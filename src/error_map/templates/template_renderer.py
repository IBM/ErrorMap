from pathlib import Path
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader


class TemplateRenderer:
    """Handles Jinja2 template loading and rendering"""
    
    def __init__(self, template_dir: Path = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "prompts"
        
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def render(self, template_name: str, **kwargs) -> str:
        """Render a Jinja2 template with given variables"""
        template = self.template_env.get_template(template_name)
        return template.render(**kwargs)
    
    def list_templates(self) -> list[str]:
        """List all available templates"""
        return self.template_env.list_templates()
    
    def template_exists(self, template_name: str) -> bool:
        """Check if a template exists"""
        try:
            self.template_env.get_template(template_name)
            return True
        except:
            return False