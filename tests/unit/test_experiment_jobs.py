"""
Unit tests for Radix Experiment Job Template Support

Tests placeholder substitution, orchestration selection, and job template building.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add cloud-api to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../services/cloud-api'))

from radix_core.experiment_jobs import (
    substitute_placeholders,
    select_orchestration,
    build_job_from_template,
    get_default_job_template,
    JobTemplateError
)


class TestPlaceholderSubstitution(unittest.TestCase):
    """Test suite for placeholder substitution"""
    
    def test_substitute_string_single_placeholder(self):
        """Test substitution in a simple string"""
        result = substitute_placeholders("--lr={{lr}}", {"lr": 0.001})
        self.assertEqual(result, "--lr=0.001")
    
    def test_substitute_string_multiple_placeholders(self):
        """Test substitution with multiple placeholders"""
        result = substitute_placeholders(
            "--lr={{lr}} --batch_size={{batch_size}}",
            {"lr": 0.001, "batch_size": 128}
        )
        self.assertEqual(result, "--lr=0.001 --batch_size=128")
    
    def test_substitute_list(self):
        """Test substitution in a list"""
        result = substitute_placeholders(
            ["python", "train.py", "--lr={{lr}}", "--batch={{batch}}"],
            {"lr": 0.01, "batch": 64}
        )
        self.assertEqual(result, ["python", "train.py", "--lr=0.01", "--batch=64"])
    
    def test_substitute_dict(self):
        """Test substitution in a dict"""
        result = substitute_placeholders(
            {"learning_rate": "{{lr}}", "epochs": "{{epochs}}"},
            {"lr": 0.001, "epochs": 10}
        )
        self.assertEqual(result, {"learning_rate": "0.001", "epochs": "10"})
    
    def test_substitute_nested_structure(self):
        """Test substitution in nested structures"""
        template = {
            "command": ["python", "train.py", "--lr={{lr}}"],
            "env": {"BATCH_SIZE": "{{batch_size}}"}
        }
        result = substitute_placeholders(template, {"lr": 0.001, "batch_size": 128})
        expected = {
            "command": ["python", "train.py", "--lr=0.001"],
            "env": {"BATCH_SIZE": "128"}
        }
        self.assertEqual(result, expected)
    
    def test_substitute_boolean_values(self):
        """Test substitution with boolean values"""
        result = substitute_placeholders("--use_cuda={{use_cuda}}", {"use_cuda": True})
        self.assertEqual(result, "--use_cuda=true")
        
        result = substitute_placeholders("--debug={{debug}}", {"debug": False})
        self.assertEqual(result, "--debug=false")
    
    def test_substitute_none_value(self):
        """Test substitution with None value"""
        result = substitute_placeholders("--value={{val}}", {"val": None})
        self.assertEqual(result, "--value=null")
    
    def test_substitute_missing_parameter_raises_error(self):
        """Test that missing parameter raises JobTemplateError"""
        with self.assertRaises(JobTemplateError) as ctx:
            substitute_placeholders("--lr={{lr}}", {})
        self.assertIn("Missing parameter", str(ctx.exception))
        self.assertIn("lr", str(ctx.exception))
    
    def test_substitute_scalar_passthrough(self):
        """Test that scalar values pass through unchanged"""
        self.assertEqual(substitute_placeholders(42, {}), 42)
        self.assertEqual(substitute_placeholders(3.14, {}), 3.14)
        self.assertEqual(substitute_placeholders(True, {}), True)
        self.assertEqual(substitute_placeholders(None, {}), None)
    
    def test_substitute_no_placeholders(self):
        """Test strings without placeholders pass through unchanged"""
        result = substitute_placeholders("python train.py", {"lr": 0.001})
        self.assertEqual(result, "python train.py")


class TestOrchestrationSelection(unittest.TestCase):
    """Test suite for orchestration selection priority"""
    
    def test_request_orchestration_priority(self):
        """Test that request orchestration has highest priority"""
        request = {"num_gpus": 4}
        config = {"num_gpus": 2}
        
        result = select_orchestration(request, config, default_num_gpus=1)
        self.assertEqual(result, {"num_gpus": 4})
    
    def test_config_orchestration_fallback(self):
        """Test that config orchestration is used when request is None"""
        config = {"num_gpus": 2, "launcher": "torchrun"}
        
        result = select_orchestration(None, config, default_num_gpus=1)
        self.assertEqual(result, {"num_gpus": 2, "launcher": "torchrun"})
    
    def test_default_orchestration_fallback(self):
        """Test that default is used when both request and config are None"""
        result = select_orchestration(None, None, default_num_gpus=1)
        self.assertEqual(result, {"num_gpus": 1})
    
    def test_custom_default_num_gpus(self):
        """Test custom default GPU count"""
        result = select_orchestration(None, None, default_num_gpus=8)
        self.assertEqual(result, {"num_gpus": 8})
    
    def test_empty_request_orchestration_falls_back_to_config(self):
        """Test that empty dict in request falls back to config (treated as falsy)"""
        request = {}
        config = {"num_gpus": 2}
        
        result = select_orchestration(request, config, default_num_gpus=1)
        self.assertEqual(result, {"num_gpus": 2})  # Empty dict is falsy, falls back to config


class TestBuildJobFromTemplate(unittest.TestCase):
    """Test suite for building jobs from templates"""
    
    def test_build_basic_job(self):
        """Test building a basic job from template"""
        template = {
            "job_kind": "training",
            "image": "pytorch:latest",
            "command": ["python", "train.py", "--lr={{lr}}"],
            "params": {"epochs": 10}
        }
        run_params = {"lr": 0.001, "batch_size": 128}
        
        result = build_job_from_template(
            template, run_params, "exp-123", "run-456"
        )
        
        self.assertEqual(result['job_kind'], "training")
        self.assertEqual(result['image'], "pytorch:latest")
        self.assertEqual(result['command'], ["python", "train.py", "--lr=0.001"])
        self.assertEqual(result['params']['epochs'], 10)
        self.assertEqual(result['params']['experiment_id'], "exp-123")
        self.assertEqual(result['params']['run_id'], "run-456")
        self.assertEqual(result['params']['run_params'], run_params)
    
    def test_build_job_with_env_substitution(self):
        """Test building job with environment variable substitution"""
        template = {
            "job_kind": "training",
            "image": "pytorch:latest",
            "command": ["python", "train.py"],
            "env": {
                "LEARNING_RATE": "{{lr}}",
                "BATCH_SIZE": "{{batch_size}}"
            }
        }
        run_params = {"lr": 0.001, "batch_size": 128}
        
        result = build_job_from_template(
            template, run_params, "exp-123", "run-456"
        )
        
        self.assertEqual(result['env']['LEARNING_RATE'], "0.001")
        self.assertEqual(result['env']['BATCH_SIZE'], "128")
    
    def test_build_job_with_args(self):
        """Test building job with args"""
        template = {
            "job_kind": "training",
            "image": "pytorch:latest",
            "command": ["python", "train.py"],
            "args": ["--config={{config}}", "--mode={{mode}}"]
        }
        run_params = {"config": "config.yaml", "mode": "train"}
        
        result = build_job_from_template(
            template, run_params, "exp-123", "run-456"
        )
        
        self.assertEqual(result['args'], ["--config=config.yaml", "--mode=train"])
    
    def test_build_job_missing_job_kind_raises_error(self):
        """Test that missing job_kind raises error"""
        template = {
            "image": "pytorch:latest",
            "command": ["python", "train.py"]
        }
        
        with self.assertRaises(JobTemplateError) as ctx:
            build_job_from_template(template, {}, "exp-123", "run-456")
        self.assertIn("job_kind", str(ctx.exception))
    
    def test_build_job_missing_image_raises_error(self):
        """Test that missing image raises error"""
        template = {
            "job_kind": "training",
            "command": ["python", "train.py"]
        }
        
        with self.assertRaises(JobTemplateError) as ctx:
            build_job_from_template(template, {}, "exp-123", "run-456")
        self.assertIn("image", str(ctx.exception))
    
    def test_build_job_invalid_template_type_raises_error(self):
        """Test that non-dict template raises error"""
        with self.assertRaises(JobTemplateError) as ctx:
            build_job_from_template("not a dict", {}, "exp-123", "run-456")
        self.assertIn("must be a dict", str(ctx.exception))
    
    def test_build_job_missing_placeholder_param_raises_error(self):
        """Test that missing placeholder parameter raises error"""
        template = {
            "job_kind": "training",
            "image": "pytorch:latest",
            "command": ["python", "train.py", "--lr={{lr}}"]
        }
        run_params = {}  # Missing 'lr'
        
        with self.assertRaises(JobTemplateError) as ctx:
            build_job_from_template(template, run_params, "exp-123", "run-456")
        self.assertIn("Missing parameter", str(ctx.exception))
    
    def test_build_job_with_params_substitution(self):
        """Test building job with params containing placeholders"""
        template = {
            "job_kind": "training",
            "image": "pytorch:latest",
            "command": ["python", "train.py"],
            "params": {
                "epochs": "{{epochs}}",
                "optimizer": "{{optimizer}}"
            }
        }
        run_params = {"epochs": 10, "optimizer": "adam"}
        
        result = build_job_from_template(
            template, run_params, "exp-123", "run-456"
        )
        
        self.assertEqual(result['params']['epochs'], "10")
        self.assertEqual(result['params']['optimizer'], "adam")
        # Experiment metadata should still be present
        self.assertEqual(result['params']['experiment_id'], "exp-123")
        self.assertEqual(result['params']['run_id'], "run-456")


class TestDefaultJobTemplate(unittest.TestCase):
    """Test suite for default job template"""
    
    def test_get_default_job_template(self):
        """Test that default template is valid"""
        template = get_default_job_template()
        
        self.assertIsInstance(template, dict)
        self.assertIn('job_kind', template)
        self.assertIn('image', template)
        self.assertEqual(template['job_kind'], 'resnet50_benchmark')
        self.assertIn('command', template)
        self.assertIn('params', template)
    
    def test_default_template_can_build_job(self):
        """Test that default template can be used to build a job"""
        template = get_default_job_template()
        
        # Should work without any run params (no placeholders in default)
        result = build_job_from_template(
            template, {}, "exp-123", "run-456"
        )
        
        self.assertEqual(result['job_kind'], 'resnet50_benchmark')
        self.assertIsNotNone(result['image'])
        self.assertIsInstance(result['command'], list)


if __name__ == '__main__':
    unittest.main()
