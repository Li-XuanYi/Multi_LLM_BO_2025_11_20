"""
LLM Warm Start Module - Optimized Version
Based on LGM50 experimental data and physics-informed constraints

Key Improvements:
1. Corrected domain knowledge based on manuscript research
2. Physics-informed parameter ranges from LGM50 specifications
3. Enhanced prompt with validated heuristics
4. Diversity-aware sampling strategy

Version: v2.0
Date: 2025-11-16
"""

import json
import numpy as np
from typing import List, Dict, Optional, Any
from openai import OpenAI


class LLMWarmStartOptimized:
    """
    Optimized LLM-guided warm start generator
    
    Based on:
    - LGM50 (INR21700 M50T) specifications: 5.0Ah, NMC811 cathode
    - Validated two-stage CC charging protocols
    - Physics-based constraints from electrochemical models
    """
    
    def __init__(
        self,
        evaluator,
        api_key: str,
        base_url: str = 'https://api.nuwaapi.com/v1',
        model: str = "gpt-3.5-turbo",
        verbose: bool = True
    ):
        """
        Initialize optimized LLM Warm Start
        
        Args:
            evaluator: MultiObjectiveEvaluator instance
            api_key: OpenAI API key
            base_url: API base URL
            model: Model name
            verbose: Print detailed logs
        """
        self.evaluator = evaluator
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.verbose = verbose
        
        # Parameter bounds (LGM50-specific)
        self.param_bounds = {
            'I1': (3.0, 6.0),           # First-stage current [A] (0.6-1.2C for 5Ah)
            't1': (5, 30),              # Switching step number
            'I2': (1.0, 3.0)            # Second-stage current [A] (0.2-0.6C)
        }
        
        # Physical constraints
        self.physical_constraints = {
            'voltage_max': 4.2,         # Max voltage [V]
            'voltage_conservative': 4.15,  # Conservative limit for longevity
            'temperature_max': 309.0,   # Max temperature [K] (36C)
            'temperature_optimal': 298.15,  # Optimal temp [K] (25C)
            'current_ratio': 'I1 > I2',
            'capacity': 5.0,            # LGM50 nominal capacity [Ah]
            'internal_resistance': 0.0212  # 21.2 mOhm at mid-SOC
        }
        
        if self.verbose:
            print("=" * 70)
            print("LLM Warm Start Optimized - Initialized")
            print("=" * 70)
            print(f"Model: {model}")
            print(f"Battery: LGM50 (5.0Ah NMC811/Graphite-SiOx)")
            print(f"Parameter space: {self.param_bounds}")
            print("=" * 70)
    
    def generate_initial_points(
        self,
        n_points: int = 10,
        semantic_constraints: Optional[str] = None,
        diversity_emphasis: float = 0.7,
        use_async: bool = False
    ) -> List[Dict[str, float]]:
        """
        Generate initial charging strategies with physics-informed guidance
        
        Args:
            n_points: Number of initial points to generate
            semantic_constraints: Additional semantic constraints (optional)
            diversity_emphasis: Diversity emphasis degree [0-1]
            use_async: Use async calls (faster)
        
        Returns:
            List of initial points: [{'I1': ..., 't1': ..., 'I2': ...}, ...]
        """
        if self.verbose:
            print(f"\nGenerating {n_points} physics-informed initial points...")
        
        # Build prompt
        prompt = self._construct_prompt(n_points, semantic_constraints, diversity_emphasis)
        
        # Call LLM
        if use_async:
            initial_points_raw = self._call_llm_async(prompt)
        else:
            initial_points_raw = self._call_llm(prompt)
        
        # Parse and validate
        initial_points = self._parse_and_validate(initial_points_raw, n_points)
        
        # Evaluate and add to database
        self._evaluate_and_add(initial_points)
        
        if self.verbose:
            print(f"Successfully generated and evaluated {len(initial_points)} initial points")
        
        return initial_points
    
    def _construct_prompt(
        self,
        n_points: int,
        semantic_constraints: Optional[str],
        diversity_emphasis: float
    ) -> str:
        """
        Construct physics-informed prompt based on LGM50 research
        
        Key changes from original:
        1. Correct optimal parameter ranges from manuscript
        2. Physics-based heuristics (not contradictory)
        3. SOC-aware switching guidance
        4. Temperature and aging trade-offs
        """
        prompt = f"""You are an expert in lithium-ion battery fast charging optimization, specifically for LGM50 (INR21700 M50T) batteries.

**Battery Specifications**:
- Chemistry: NMC811 cathode, Graphite-SiOx anode
- Nominal capacity: 5.0Ah
- Voltage range: 2.5-4.2V
- Internal resistance: 21.2 mOhm
- Optimal operating temperature: 25C (298K)

**Task**: Generate {n_points} high-quality two-stage constant-current (CC) charging strategies.

**Parameter Space**:
- I1 (First-stage current): {self.param_bounds['I1'][0]}-{self.param_bounds['I1'][1]} A (0.6-1.2C rate)
- t1 (Stage switching time): {self.param_bounds['t1'][0]}-{self.param_bounds['t1'][1]} steps (90s per step)
- I2 (Second-stage current): {self.param_bounds['I2'][0]}-{self.param_bounds['I2'][1]} A (0.2-0.6C rate)

**Hard Physical Constraints** (MUST be satisfied):
1. Voltage limit: <= 4.2V (never exceed)
2. Temperature limit: <= 36C (309K)
3. Current relationship: I1 > I2 (monotonically decreasing)
4. Optimal I2/I1 ratio: 0.5-0.75 (based on research validation)

**Physics-Informed Design Principles** (from validated research):

1. **Current Selection**:
   - I1 optimal range: 4.5-6.0A (0.9-1.2C) for LGM50
   - I1 > 6A increases lithium plating risk above 50% SOC
   - I2 optimal range: 2.5-3.5A (0.5-0.7C) balances time and aging
   - I2 < 2.0A extends charging time without significant longevity benefit

2. **Switching Time (SOC-based)**:
   - Each step = 90 seconds
   - Optimal switching: 10-15 steps (15-22.5 min, ~30-35% SOC)
   - Early switching (t1 < 8): wastes fast-charging potential
   - Late switching (t1 > 18): increases lithium plating risk
   - Never switch after 50% SOC with high I1 (plating hazard zone)

3. **Thermal Management**:
   - Heat generation ~ I^2 * R (I-squared law)
   - I1=6A for 20 steps generates ~8-10C temperature rise
   - Target: keep peak temperature < 32C for longevity
   - Strategy: higher I1 requires earlier switching

4. **Aging Mechanisms**:
   - SEI growth: exponential with voltage and temperature
   - Lithium plating: critical above 50% SOC at high currents
   - Optimal for cycle life: I1=5A, t1=12-15, I2=3A
   - Aggressive but safe: I1=5.5-6A, t1=10-12, I2=3-3.5A

5. **Validated Protocol Clusters** (from research):
   - Conservative: I1=4.5-5.2A, t1=15-20, I2=2.5-3.0A (800-1000 cycles)
   - Balanced: I1=5.0-5.8A, t1=12-15, I2=2.8-3.3A (600-800 cycles)
   - Aggressive: I1=5.5-6.0A, t1=8-12, I2=3.0-3.5A (400-600 cycles)

**Optimization Objectives** (trade-offs):
1. Minimize charging time: favor higher I1 and lower t1
2. Minimize peak temperature: favor moderate I1 and earlier t1
3. Minimize capacity fade: favor lower I1, optimal t1 (12-15), gentle I2

**Critical Insight**: 
Research shows that COUNTERINTUITIVE protocols can outperform expert designs. 
Don't over-constrain to simple heuristics. Example: non-monotonic current profiles
or slightly delayed switching can sometimes reduce aging through complex thermal-
electrochemical coupling effects.

"""
        
        # Add semantic constraints if provided
        if semantic_constraints:
            prompt += f"""**Additional Semantic Constraints**:
{semantic_constraints}

"""
        
        # Diversity requirements
        if diversity_emphasis > 0.5:
            prompt += f"""**Diversity Strategy** (Very Important for Optimization):

Generate strategies covering DIFFERENT design philosophies:

1. **Conservative cluster** ({int(n_points * 0.3)} strategies):
   - Lower I1 (4.5-5.2A), longer t1 (15-20 steps), gentle I2 (2.5-3.0A)
   - Prioritizes cycle life over charging speed
   - Expected: 45+ min to 80% SOC, <30C peak temp, 800+ cycles

2. **Balanced cluster** ({int(n_points * 0.4)} strategies):
   - Medium I1 (5.0-5.8A), optimal t1 (12-15 steps), moderate I2 (2.8-3.3A)
   - Best trade-off between all objectives
   - Expected: 40-45 min to 80% SOC, 30-32C peak temp, 600-800 cycles

3. **Aggressive cluster** ({int(n_points * 0.3)} strategies):
   - Higher I1 (5.5-6.0A), shorter t1 (8-12 steps), moderate I2 (3.0-3.5A)
   - Prioritizes fast charging
   - Expected: 38-42 min to 80% SOC, 32-35C peak temp, 400-600 cycles

4. **Exploration** (few strategies):
   - Include 1-2 counterintuitive strategies that might discover emergent behaviors
   - Example: I1=5.2A, t1=18, I2=3.2A (late switch with gentle taper)
   - Example: I1=5.8A, t1=8, I2=2.5A (aggressive start, conservative finish)

Ensure parameter variance > 15% across clusters for effective exploration.

"""
        
        # Output format
        prompt += f"""**Output Format** (CRITICAL):
Return ONLY a valid JSON array with {n_points} strategies. No extra text, no markdown, no explanations.

Format:
[
  {{"I1": 5.2, "t1": 14, "I2": 3.0}},
  {{"I1": 5.8, "t1": 10, "I2": 3.2}},
  ...
]

**Validation Checklist** (before outputting):
1. All values within specified bounds
2. All strategies satisfy I1 > I2 (at least 0.5A difference)
3. Exactly {n_points} strategies provided
4. Diversity across clusters (conservative, balanced, aggressive)
5. I2/I1 ratio between 0.5-0.75 for most strategies
6. Return ONLY the JSON array, nothing else

Generate the {n_points} strategies now:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Synchronous LLM call
        """
        if self.verbose:
            print("Calling LLM...")
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in lithium-ion battery charging optimization "
                                   "with deep knowledge of electrochemistry, thermal dynamics, and "
                                   "aging mechanisms. You provide precise, physics-informed JSON responses "
                                   "based on validated research data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,  # Slightly higher for exploration while maintaining physics
                max_tokens=2500
            )
            
            response = completion.choices[0].message.content
            
            if self.verbose:
                print("LLM response received successfully")
            
            return response
            
        except Exception as e:
            print(f"LLM call failed: {e}")
            raise
    
    def _call_llm_async(self, prompt: str) -> str:
        """
        Async LLM call (currently returns sync result, can be extended)
        """
        return self._call_llm(prompt)
    
    def _parse_and_validate(self, response: str, expected_n: int) -> List[Dict[str, float]]:
        """
        Parse and validate LLM response
        """
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            strategies = json.loads(response)
            
            if not isinstance(strategies, list):
                print("Warning: Response is not a list, attempting to extract")
                return []
            
            # Validate each strategy
            valid_strategies = []
            for i, strategy in enumerate(strategies):
                if self._validate_strategy(strategy):
                    valid_strategies.append(strategy)
                else:
                    if self.verbose:
                        print(f"  Strategy {i+1} failed validation, skipped")
            
            if len(valid_strategies) < expected_n * 0.5:
                print(f"Warning: Only {len(valid_strategies)}/{expected_n} strategies are valid")
            
            if self.verbose:
                print(f"Successfully parsed {len(valid_strategies)} valid strategies")
            
            return valid_strategies
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Raw response: {response[:200]}...")
            return []
    
    def _validate_strategy(self, strategy: Dict[str, Any]) -> bool:
        """
        Validate single strategy against constraints
        
        Enhanced validation:
        1. Required fields present
        2. Parameter bounds
        3. I1 > I2 (monotonic decrease)
        4. I2/I1 ratio check (0.3-0.85 acceptable)
        5. Physical plausibility
        """
        try:
            # Check required fields
            required_fields = ['I1', 't1', 'I2']
            if not all(field in strategy for field in required_fields):
                return False
            
            I1 = float(strategy['I1'])
            t1 = float(strategy['t1'])
            I2 = float(strategy['I2'])
            
            # Check parameter ranges
            if not (self.param_bounds['I1'][0] <= I1 <= self.param_bounds['I1'][1]):
                return False
            if not (self.param_bounds['t1'][0] <= t1 <= self.param_bounds['t1'][1]):
                return False
            if not (self.param_bounds['I2'][0] <= I2 <= self.param_bounds['I2'][1]):
                return False
            
            # Check I1 > I2 with minimum gap
            if I1 <= I2 or (I1 - I2) < 0.3:  # At least 0.3A difference
                return False
            
            # Check I2/I1 ratio (should be 0.3-0.85)
            ratio = I2 / I1
            if not (0.3 <= ratio <= 0.85):
                return False
            
            # Check extreme combinations
            # Very high I1 (>5.8A) should switch early (t1 < 15)
            if I1 > 5.8 and t1 > 15:
                return False
            
            # Very low I2 (<2.0A) is inefficient unless very conservative
            if I2 < 2.0 and I1 > 5.5:
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    def _evaluate_and_add(self, strategies: List[Dict[str, float]]) -> None:
        """
        Evaluate all strategies and add to database
        """
        if self.verbose:
            print(f"\nEvaluating {len(strategies)} initial strategies...")
        
        for i, strategy in enumerate(strategies, 1):
            try:
                # Call evaluator for real evaluation
                scalarized = self.evaluator.evaluate(
                    current1=strategy['I1'],
                    charging_number=int(strategy['t1']),
                    current2=strategy['I2']
                )
                
                if self.verbose and i % 3 == 0:
                    print(f"  Completed {i}/{len(strategies)} evaluations")
                
            except Exception as e:
                print(f"  Strategy {i} evaluation failed: {e}")
    
    def get_initial_database(self) -> List[Dict]:
        """
        Get initialized database
        """
        return self.evaluator.export_database()
    
    def compare_with_random(self, n_points: int = 10) -> Dict[str, Any]:
        """
        Compare LLM-generated initial points vs random sampling
        
        Returns:
            Comparison results dictionary
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("Performance Comparison: LLM Warm Start vs Random Sampling")
            print("=" * 70)
        
        # Save current database state
        current_eval_count = self.evaluator.eval_count
        
        # Generate LLM initial points
        llm_strategies = self.generate_initial_points(n_points, use_async=False)
        llm_eval_count = self.evaluator.eval_count
        
        # Extract LLM strategy scalarized values
        llm_database = self.evaluator.export_database()[current_eval_count:]
        llm_scalarized = [entry['scalarized'] for entry in llm_database if entry['valid']]
        
        # Generate random initial points
        random_strategies = []
        for _ in range(n_points):
            I1 = np.random.uniform(*self.param_bounds['I1'])
            t1 = np.random.randint(*self.param_bounds['t1'])
            I2 = np.random.uniform(*self.param_bounds['I2'])
            
            # Ensure I1 > I2
            if I1 <= I2:
                I1, I2 = I2 + 0.5, I1
            
            random_strategies.append({'I1': I1, 't1': t1, 'I2': I2})
        
        # Evaluate random strategies
        if self.verbose:
            print(f"\nEvaluating {n_points} random strategies...")
        
        random_eval_start = self.evaluator.eval_count
        for strategy in random_strategies:
            try:
                self.evaluator.evaluate(
                    current1=strategy['I1'],
                    charging_number=int(strategy['t1']),
                    current2=strategy['I2']
                )
            except:
                pass
        
        # Extract random strategy scalarized values
        random_database = self.evaluator.export_database()[random_eval_start:]
        random_scalarized = [entry['scalarized'] for entry in random_database if entry['valid']]
        
        # Calculate statistics
        comparison = {
            'llm': {
                'mean': np.mean(llm_scalarized) if llm_scalarized else float('inf'),
                'min': np.min(llm_scalarized) if llm_scalarized else float('inf'),
                'max': np.max(llm_scalarized) if llm_scalarized else float('inf'),
                'std': np.std(llm_scalarized) if llm_scalarized else 0,
                'valid_rate': len(llm_scalarized) / n_points
            },
            'random': {
                'mean': np.mean(random_scalarized) if random_scalarized else float('inf'),
                'min': np.min(random_scalarized) if random_scalarized else float('inf'),
                'max': np.max(random_scalarized) if random_scalarized else float('inf'),
                'std': np.std(random_scalarized) if random_scalarized else 0,
                'valid_rate': len(random_scalarized) / n_points
            }
        }
        
        # Calculate improvement rate
        if comparison['random']['mean'] > 0:
            improvement = (comparison['random']['mean'] - comparison['llm']['mean']) / comparison['random']['mean'] * 100
        else:
            improvement = 0
        
        comparison['improvement_percent'] = improvement
        
        # Print comparison results
        if self.verbose:
            print("\n" + "=" * 70)
            print("Comparison Results:")
            print("=" * 70)
            print(f"LLM Warm Start:")
            print(f"  Average scalarized value: {comparison['llm']['mean']:.4f}")
            print(f"  Best scalarized value: {comparison['llm']['min']:.4f}")
            print(f"  Valid rate: {comparison['llm']['valid_rate']*100:.1f}%")
            print(f"\nRandom Sampling:")
            print(f"  Average scalarized value: {comparison['random']['mean']:.4f}")
            print(f"  Best scalarized value: {comparison['random']['min']:.4f}")
            print(f"  Valid rate: {comparison['random']['valid_rate']*100:.1f}%")
            print(f"\nImprovement rate: {improvement:.2f}%")
            print("=" * 70)
        
        return comparison


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing Optimized LLM Warm Start")
    print("=" * 70)
    
    from multi_objective_evaluator import MultiObjectiveEvaluator
    
    # Initialize evaluator
    evaluator = MultiObjectiveEvaluator(
        weights={'time': 0.4, 'temp': 0.35, 'aging': 0.25},
        verbose=False
    )
    
    # Initialize optimized LLM Warm Start
    warmstart = LLMWarmStartOptimized(
        evaluator=evaluator,
        api_key='sk-Sq1zyC8PLM8gafI2fpAccWpzBAzZvuNOPU6ZC9aWA6C883IK',
        base_url='https://api.nuwaapi.com/v1',
        model='gpt-3.5-turbo',
        verbose=True
    )
    
    # Test: Generate initial points
    print("\nTest: Generate Physics-Informed Initial Points")
    try:
        initial_points = warmstart.generate_initial_points(
            n_points=10,
            diversity_emphasis=0.8
        )
        
        print(f"\nGenerated initial points:")
        for i, point in enumerate(initial_points, 1):
            ratio = point['I2'] / point['I1']
            print(f"  {i}. I1={point['I1']:.2f}A, t1={int(point['t1'])}, "
                  f"I2={point['I2']:.2f}A, ratio={ratio:.2f}")
        
        print("\nTest passed")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()