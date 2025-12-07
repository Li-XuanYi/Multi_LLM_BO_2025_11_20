"""
PyBaMM Sensitivity Computer v3.0
统一的灵敏度计算接口

=============================================================================
改进内容
=============================================================================
**v3.0 重大升级**：
1. 支持两种灵敏度计算模式：
   - 'finite_difference': 改进的有限差分（带惩罚梯度）
   - 'auto_diff': PyBaMM原生自动微分（规划中）

2. 智能梯度处理：
   - 约束违反时返回"指向可行域"的惩罚梯度
   - 梯度不再为零，耦合矩阵能正确捕捉边界物理

3. 性能提升：
   - 有限差分：保持 7 次仿真，但梯度质量更高
   - 自动微分：1 次仿真，5-10 倍加速（规划中）

=============================================================================
使用示例
=============================================================================
```python
# 创建计算器（推荐配置）
computer = PyBaMMSensitivityComputer(
    mode='finite_difference',
    enable_penalty_gradients=True,
    verbose=True
)

# 计算多目标梯度
params = {'current1': 5.0, 'charging_number': 10, 'current2': 3.0}
gradients = computer.compute_multi_objective_gradients(params)

# 即使在约束边界，梯度也不为零
print(gradients['temp']['current1'])  # 非零，指向降温方向
```

=============================================================================
作者: Research Team
日期: 2025-01-20
版本: v3.0 - Penalty Gradients & Dual Modes
=============================================================================
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Literal

try:
    from SPM_v3 import SPM_Sensitivity
except ImportError:
    try:
        from .SPM_v3 import SPM_Sensitivity
    except ImportError:
        try:
            from SPM import SPM_Sensitivity
            warnings.warn(
                "Using old SPM version. Consider upgrading to SPM_v3 for penalty gradients.",
                UserWarning
            )
        except ImportError:
            raise ImportError("Cannot import SPM module")


class PyBaMMSensitivityComputer:
    """
    统一的灵敏度计算接口
    
    封装 SPM_v3 的双模式功能，提供简洁的 API
    """
    
    def __init__(
        self,
        mode: Literal['finite_difference', 'auto_diff'] = 'finite_difference',
        enable_penalty_gradients: bool = True,
        penalty_scale: float = 10.0,
        verbose: bool = True
    ):
        """
        初始化灵敏度计算器
        
        参数：
            mode: 灵敏度计算模式
                  'finite_difference': 改进的有限差分（带惩罚梯度）
                  'auto_diff': PyBaMM原生自动微分（规划中）
            enable_penalty_gradients: 是否启用惩罚梯度
            penalty_scale: 惩罚梯度的缩放系数
            verbose: 是否打印详细信息
        """
        self.mode = mode
        self.enable_penalty_gradients = enable_penalty_gradients
        self.penalty_scale = penalty_scale
        self.verbose = verbose
        
        self.param_names = ['current1', 'charging_number', 'current2']
        
        if self.verbose:
            print("\n[PyBaMMSensitivityComputer v3.0]")
            print(f"  Mode: {mode}")
            if mode == 'finite_difference':
                print(f"  Penalty Gradients: {'ENABLED' if enable_penalty_gradients else 'DISABLED'}")
                if enable_penalty_gradients:
                    print(f"  Penalty Scale: {penalty_scale}")
                    print("  [OK] Constraint violations will return gradients pointing to feasible region")
    
    def compute_multi_objective_gradients(
        self,
        params: Dict[str, float],
        objectives: List[str] = None,
        spm_instance: Optional[SPM_Sensitivity] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        计算多目标梯度
        
        ∇f = [dtime/dθ, dtemp/dθ, daging/dθ]
        
        **v3.0 改进**：即使在约束边界，梯度也不为零
        
        参数：
            params: {'current1': ..., 'charging_number': ..., 'current2': ...}
            objectives: 要计算梯度的目标列表，默认 ['time', 'temp', 'aging']
            spm_instance: SPM_Sensitivity 实例（可选，用于复用）
        
        返回：
            gradients: {
                'time': {'current1': ..., 'charging_number': ..., 'current2': ...},
                'temp': {...},
                'aging': {...}
            }
        """
        if objectives is None:
            objectives = ['time', 'temp', 'aging']
        
        # 创建或复用 SPM 实例
        if spm_instance is None:
            spm = SPM_Sensitivity(
                mode=self.mode,
                enable_penalty_gradients=self.enable_penalty_gradients,
                penalty_scale=self.penalty_scale,
                verbose=False
            )
        else:
            spm = spm_instance
        
        try:
            # 运行仿真并获取灵敏度
            result = spm.run_two_stage_charging(
                current1=params['current1'],
                charging_number=int(params['charging_number']),
                current2=params['current2'],
                return_sensitivities=True
            )
            
            if not result['valid']:
                if self.verbose:
                    violations = result.get('constraint_violations', {})
                    if violations:
                        print(f"  [Warning] Constraint violations detected:")
                        for constraint, info in violations.items():
                            if isinstance(info, dict) and 'excess' in info:
                                print(f"    - {constraint}: {info['value']:.2f} (limit={info['limit']:.2f}, excess={info['excess']:.2f})")
                
                # **关键改进**：即使违反约束，仍然返回梯度
                if 'sensitivities' in result and result['sensitivities']:
                    if self.verbose:
                        print(f"  [OK] Penalty gradients returned (pointing to feasible region)")
                        sample_grad = result['sensitivities']['temp']['current1']
                        print(f"    Example: dtemp/dI1 = {sample_grad:.4f}")
                    return result['sensitivities']
                else:
                    warnings.warn("No sensitivities returned, returning zero gradients")
                    return self._zero_gradients(objectives)
            
            # 提取灵敏度
            if 'sensitivities' in result:
                gradients = result['sensitivities']
                
                if self.verbose:
                    print(f"  [OK] Gradients computed successfully")
                
                return gradients
            else:
                warnings.warn("No sensitivities returned, returning zero gradients")
                return self._zero_gradients(objectives)
        
        except Exception as e:
            warnings.warn(f"Gradient computation failed: {e}")
            return self._zero_gradients(objectives)
    
    def compute_jacobian(
        self,
        params: Dict[str, float],
        spm_instance: Optional[SPM_Sensitivity] = None
    ) -> np.ndarray:
        """
        计算雅可比矩阵（所有目标对所有参数的梯度）
        
        J[i,j] = dobjective_i / dparam_j
        
        返回：
            J: (3, 3) 雅可比矩阵
               行: [time, temp, aging]
               列: [current1, charging_number, current2]
        """
        gradients = self.compute_multi_objective_gradients(
            params,
            objectives=['time', 'temp', 'aging'],
            spm_instance=spm_instance
        )
        
        # 构建雅可比矩阵
        J = np.zeros((3, 3))
        
        objectives = ['time', 'temp', 'aging']
        for i, obj in enumerate(objectives):
            for j, param in enumerate(self.param_names):
                J[i, j] = gradients[obj][param]
        
        if self.verbose:
            print("\n[Jacobian Matrix]")
            print("           I1        t1        I2")
            for i, obj in enumerate(objectives):
                print(f"  {obj:6s}  {J[i,0]:8.4f}  {J[i,1]:8.4f}  {J[i,2]:8.4f}")
        
        return J
    
    def estimate_parameter_sensitivity(
        self,
        params: Dict[str, float],
        spm_instance: Optional[SPM_Sensitivity] = None
    ) -> Dict[str, str]:
        """
        估计参数敏感度等级
        
        返回：
            {'current1': 'high'|'medium'|'low', ...}
        """
        J = self.compute_jacobian(params, spm_instance)
        
        # 计算每个参数的总影响（所有目标梯度的 L2 范数）
        param_influence = {}
        for j, param in enumerate(self.param_names):
            influence = np.linalg.norm(J[:, j])
            param_influence[param] = influence
        
        # 归一化到 [0, 1]
        max_influence = max(param_influence.values())
        if max_influence > 1e-10:
            normalized = {
                p: v / max_influence
                for p, v in param_influence.items()
            }
        else:
            normalized = {p: 0.0 for p in param_influence}
        
        # 分级
        sensitivity = {}
        for param, norm_val in normalized.items():
            if norm_val > 0.6:
                sensitivity[param] = 'high'
            elif norm_val > 0.3:
                sensitivity[param] = 'medium'
            else:
                sensitivity[param] = 'low'
        
        if self.verbose:
            print("\n[Parameter Sensitivity]")
            for param, level in sensitivity.items():
                print(f"  {param}: {level} (normalized={normalized[param]:.3f})")
        
        return sensitivity
    
    def _zero_gradients(self, objectives: List[str]) -> Dict[str, Dict[str, float]]:
        """返回零梯度"""
        gradients = {}
        for obj in objectives:
            gradients[obj] = {param: 0.0 for param in self.param_names}
        return gradients


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing PyBaMMSensitivityComputer v3.0")
    print("=" * 70)
    
    print("\n[Test 1] Normal point")
    test_params = {
        'current1': 4.5,
        'charging_number': 15,
        'current2': 3.0
    }
    
    computer = PyBaMMSensitivityComputer(
        mode='finite_difference',
        enable_penalty_gradients=True,
        verbose=True
    )
    
    gradients = computer.compute_multi_objective_gradients(test_params)
    
    print("\n[Test 2] Boundary point (high current)")
    boundary_params = {
        'current1': 5.8,
        'charging_number': 20,
        'current2': 3.8
    }
    
    gradients_boundary = computer.compute_multi_objective_gradients(boundary_params)
    
    print("\n" + "=" * 70)
    print("Testing completed")
    print("=" * 70)
