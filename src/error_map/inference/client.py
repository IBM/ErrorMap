from typing import Any, Dict, Optional
import os
import asyncio
import aiohttp
import litellm
from error_map.templates.json_renderer import JSONRenderer
from ..templates import TemplateRenderer
from litellm.caching.caching import Cache


class InferenceClient:
    def __init__(
        self,
        inference_type: Optional[str] = "litellm-mock",
        judge: Optional[str] = None,
        provider: Optional[str] = None,
        max_workers: int = None,
        litellm_config: Optional[Dict] = None,
    ):
        """
        LLM client on top of LiteLLM.

        Currently supported providers: Azure, Rits.
        You must either select one of the supported providers or provide a `litellm_config` with all the required parameters for your chosen provider.
        """
        self.inference_type = inference_type.lower() if inference_type else None
        self.provider = provider or "rits"
        self.max_workers = max_workers
        self.litellm_config = litellm_config
        self.semaphore = asyncio.Semaphore(max_workers)
        self.template_renderer = TemplateRenderer()
        self.schema_renderer = JSONRenderer()
        
        self.client = litellm
        litellm.cache = Cache(type="disk", path="./litellm_cache")    

        if litellm_config:
            self.judge = litellm_config.get("model", "")
            self.api_base = litellm_config.get("api_base", "")
            self.api_key = litellm_config.get("api_key", "")

        elif self.provider == "azure":
            judge = judge or "Azure/gpt-4.1"
            self.api_base = os.getenv("AZURE_API_BASE")
            self.api_key = os.getenv("AZURE_API_KEY")
            self.judge = self._normalize_model(judge)

        elif self.provider == "rits":
            model_name_to_endpoint_name = {
                "Qwen/Qwen2.5-72B-Instruct": "qwen2-5-72b-instruct",
                "deepseek-ai/DeepSeek-V2.5": "deepseek-v2-5",
                "openai/gpt-oss-120b": "gpt-oss-120b",
                "deepseek-ai/DeepSeek-V3": "deepseek-v3-h200",
                "Qwen/Qwen3-30B-A3B-Thinking-2507": "qwen3-30b-a3b-thinking-2507",
            }

            judge = judge or "openai/gpt-oss-120b"
            self.api_base = f"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_name_to_endpoint_name[judge]}/v1"
            self.api_key = os.getenv("RITS_API_KEY")
            self.judge = self._normalize_model(judge)
        
        else:
            raise Exception("Neither a LiteLLM config nor a valid provider was provided!")
       

    def _normalize_model(self, model: str) -> str:
        """
        Normalize model name based on provider.
        """
        if self.provider == "azure":
            if not model.startswith("azure/"):
                return f"azure/{model}"
        elif self.provider == "rits":
            return f"openai/{model}"
        return model

    async def infer(
        self,
        template_name: str,
        template_vars: Dict[str, Any],
        schema_name: str = "",
        timeout: float = 1000.0,  # max seconds per infer
        max_tokens: int = 10000,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async call to LLM with worker control.
        """
        prompt = self.render_prompt(template_name, **template_vars)
        message = [{"role": "user", "content": prompt}]

        infer_params = {
            "model": self.judge,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "messages": message,
            "timeout": timeout,
            "max_tokens": max_tokens,
            "extra_headers": {'RITS_API_KEY': self.api_key} if self.provider == "rits" else None,
        }

        if schema_name:
            schema = self.render_schema(schema_name)
            if schema is not None:
                infer_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "schema", "schema": schema},
                }

        if self.litellm_config:
            infer_params.update(self.litellm_config)

        async with self.semaphore: # worker limit
            
            if self.inference_type == "litellm-mock":
                return {
                    "model": self.judge,
                    "prompt": prompt,
                    "template": template_name,
                    "success": True,
                    "full_response": "mock response",
                    "content": "mock response content",
                }
            
            try:
                response = await self.client.acompletion(
                        **infer_params,
                        **kwargs
                    )
                
            except Exception as e:
                print("EXCEPTION: ", e)
                return {
                    "model": self.judge,
                    "prompt": prompt,
                    "template": template_name,
                    "success": False,
                    "error": str(e),
                    "full_response": None,
                    "content": None,
                }

        return {
            "model": self.judge,
            "prompt": prompt,
            "template": template_name,
            "success": True,
            "full_response": response,
            "content": response.choices[0].message.content,
        }
    
    def render_prompt(self, template_name: str, **kwargs) -> str:
        """Render a Jinja2 template with given variables"""
        return self.template_renderer.render(template_name, **kwargs)
    
    def render_schema(self, schema_name: str) -> Any:
        return self.schema_renderer.render(schema_name)

    async def __aenter__(self):
        # open session once
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # close session when done
        await self.session.close()