
import functools
import pandas as pd
from pathlib import Path
from typing import List, Dict, Callable, Optional


def cached(stage_name: str, output_path: Optional[Path]):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> List[Dict]:
            # Get exp_id from kwargs
            exp_id = kwargs.get('exp_id')
            if not exp_id:
                raise ValueError(f"Function {func.__name__} must have 'exp_id' parameter")
            
            # Generate cache filename exactly like original
            cache_kwargs = {k: v for k, v in kwargs.items() if k != 'exp_id'}
            filename_parts = [f"exp_name={stage_name}", f"exp_id={exp_id}"]
            
            # for key, value in cache_kwargs.items():
            #     if value is not None and key in ['models', 'ratio', 'seed', 'field']:
            #         if isinstance(value, list):
            #             value = '_'.join(map(str, value))
            #         filename_parts.append(f"{key}={value}")
            
            filename = "__".join(filename_parts) + ".csv"
            if output_path is None:
                raise ValueError("output_path cannot be None")
            cache_path = output_path / filename
            
            # Try to load from cache
            if cache_path.exists():
                try:
                    df = pd.read_csv(cache_path)
                    print(f"📁 Using cached {stage_name} results ({len(df)} records)")
                    return df.to_dict('records')
                except Exception as e:
                    print(f"⚠️ Failed to load cache, regenerating: {e}")
            
            # Execute function
            print(f"🔄 Running {stage_name}...")
            results = await func(*args, **kwargs)
            
            # Save to cache with exact backward compatible format
            try:
                df = pd.DataFrame(results)
                
                # Fix data types for exact compatibility 
                if 'score' in df.columns:
                    df['score'] = pd.to_numeric(df['score'], errors='coerce')
                
                # Ensure exact column order for data_preparation
                if stage_name == "data_preparation":
                    original_columns = ['index', 'example_id', 'potential_answers', 'references', 'model', 
                                       'output_text', 'score', 'input_text', 'correct_answer', 'dataset', 
                                       'dataset_category', 'prediction', 'error']
                    available_columns = [col for col in original_columns if col in df.columns]
                    extra_columns = [col for col in df.columns if col not in original_columns]
                    df = df[available_columns + extra_columns]
                
                df.to_csv(cache_path, index=False)
                print(f"💾 Cached {stage_name} results ({len(results)} records)")
            except Exception as e:
                print(f"⚠️ Failed to cache results: {e}")
            
            return results
        
        return wrapper
    return decorator