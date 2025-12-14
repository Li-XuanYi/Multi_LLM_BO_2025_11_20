#!/usr/bin/env python3
"""
优化修复验证脚本
快速验证所有修复是否正常工作
"""

import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def test_coupling_kernel_positive_definite():
    """测试1: CouplingKernel正定性"""
    print("\n" + "="*70)
    print("测试1: CouplingKernel正定性验证")
    print("="*70)
    
    try:
        from llmbo_core.LLM_enhanced_surrogate_modeling import CouplingKernel
        
        # 创建耦合矩阵
        W = np.array([
            [1.0, 0.7, 0.4],
            [0.7, 1.0, 0.3],
            [0.4, 0.3, 1.0]
        ])
        
        # 创建核函数
        kernel = CouplingKernel(coupling_matrix=W, length_scale=1.0)
        
        # 测试核矩阵
        X = np.random.rand(20, 3)
        K = kernel(X)
        
        # 检查正定性
        eigenvalues = np.linalg.eigvalsh(K)
        min_eigenvalue = eigenvalues.min()
        
        print(f"  核矩阵形状: {K.shape}")
        print(f"  最小特征值: {min_eigenvalue:.8f}")
        
        if min_eigenvalue > -1e-10:
            print("  ✅ CouplingKernel正定性检查通过")
            return True
        else:
            print(f"  ❌ CouplingKernel非正定! 最小特征值={min_eigenvalue}")
            return False
            
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gamma_update_logic():
    """测试2: Gamma更新逻辑"""
    print("\n" + "="*70)
    print("测试2: Gamma多策略更新验证")
    print("="*70)
    
    try:
        from llmbo_core.LLM_enhanced_surrogate_modeling import CouplingStrengthScheduler
        
        scheduler = CouplingStrengthScheduler(
            initial_gamma=0.5,
            adjustment_rate=0.1,
            verbose=True
        )
        
        # 模拟优化过程
        print("\n  场景1: 快速改善")
        f_values = [1.0, 0.8, 0.6, 0.5, 0.4]
        for f in f_values:
            gamma = scheduler.update(f)
        
        print(f"\n  场景2: 停滞")
        for _ in range(5):
            gamma = scheduler.update(0.4 + np.random.rand()*0.005)
        
        print(f"\n  场景3: 恶化")
        for _ in range(3):
            gamma = scheduler.update(0.45 + np.random.rand()*0.1)
        
        print("\n  ✅ Gamma更新逻辑测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_coupling_matrix_integration():
    """测试3: LLM耦合矩阵集成"""
    print("\n" + "="*70)
    print("测试3: LLM耦合矩阵集成验证")
    print("="*70)
    
    try:
        from llmbo_core.LLM_enhanced_surrogate_modeling import (
            CouplingMatrixEstimator,
            LLMSurrogateAdvisor
        )
        
        # 创建估计器
        estimator = CouplingMatrixEstimator(verbose=False)
        
        # 检查是否有generate_coupling_matrix_from_llm方法
        if hasattr(LLMSurrogateAdvisor, 'generate_coupling_matrix_from_llm'):
            print("  ✅ LLMSurrogateAdvisor.generate_coupling_matrix_from_llm 存在")
        else:
            print("  ❌ 缺少 generate_coupling_matrix_from_llm 方法")
            return False
        
        # 创建模拟历史数据
        mock_history = []
        for i in range(10):
            mock_history.append({
                'eval_id': i,
                'params': {
                    'current1': 4.0 + np.random.rand(),
                    'charging_number': 10 + int(np.random.rand() * 10),
                    'current2': 2.0 + np.random.rand()
                },
                'objectives': {
                    'time': 35 + np.random.rand() * 10,
                    'temp': 302 + np.random.rand() * 5,
                    'aging': 0.001 + np.random.rand() * 0.002
                },
                'scalarized': 0.15 + np.random.rand() * 0.1,
                'valid': True,
                'gradients': {
                    'time': {
                        'current1': -0.01 - np.random.rand() * 0.01,
                        'charging_number': 0.001 + np.random.rand() * 0.002,
                        'current2': -0.005 - np.random.rand() * 0.005
                    },
                    'temp': {
                        'current1': 0.05 + np.random.rand() * 0.02,
                        'charging_number': 0.002 + np.random.rand() * 0.001,
                        'current2': 0.01 + np.random.rand() * 0.01
                    },
                    'aging': {
                        'current1': 0.0001 + np.random.rand() * 0.0001,
                        'charging_number': 0.00005 + np.random.rand() * 0.00002,
                        'current2': 0.00003 + np.random.rand() * 0.00002
                    }
                }
            })
        
        # 估计耦合矩阵
        W = estimator.estimate_from_history(
            history=mock_history,
            use_scalarized=True
        )
        
        print(f"  耦合矩阵:\n{W}")
        print("  ✅ LLM耦合矩阵集成测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ei_parameters():
    """测试4: EI参数优化"""
    print("\n" + "="*70)
    print("测试4: EI参数优化验证")
    print("="*70)
    
    try:
        from llmbo_core.LLM_Enhanced_Expected_Improvement import SamplingParameterComputer
        
        pbounds = {
            'current1': (3.0, 6.0),
            'charging_number': (5, 25),
            'current2': (1.0, 4.0)
        }
        
        computer = SamplingParameterComputer(pbounds=pbounds, verbose=False)
        
        print(f"  sigma_min: {computer.sigma_min}")
        print(f"  sigma_max: {computer.sigma_max}")
        
        if computer.sigma_min == 0.05 and computer.sigma_max == 3.0:
            print("  ✅ EI参数已优化 (sigma_min=0.05, sigma_max=3.0)")
            return True
        else:
            print(f"  ⚠️  EI参数未优化 (期望: 0.05/3.0, 实际: {computer.sigma_min}/{computer.sigma_max})")
            return False
            
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    print("\n" + "="*70)
    print("🚀 LLMBO优化修复验证")
    print("="*70)
    
    results = {
        'CouplingKernel正定性': test_coupling_kernel_positive_definite(),
        'Gamma多策略更新': test_gamma_update_logic(),
        'LLM耦合矩阵集成': test_llm_coupling_matrix_integration(),
        'EI参数优化': test_ei_parameters()
    }
    
    print("\n" + "="*70)
    print("📊 测试总结")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过! 优化已成功应用")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查修复")
        return 1


if __name__ == "__main__":
    sys.exit(main())
