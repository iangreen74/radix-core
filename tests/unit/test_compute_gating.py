"""
Unit tests for compute gating logic.
"""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


def test_check_compute_access_design_partner():
    """Test that design partners are allowed without billing."""
    from handlers import check_compute_access
    
    tenant = {
        'tenant_id': 'tenant-123',
        'plan': 'design-partner',
        'subscription_status': 'PENDING',
        'plan_code': None,
        'licensed_gpu_count': 0,
        'trial_status': 'NONE',
    }
    
    result = check_compute_access(tenant)
    assert result is None  # Allowed


def test_check_compute_access_active_subscription():
    """Test that active subscription with licensed GPUs is allowed."""
    from handlers import check_compute_access
    
    tenant = {
        'tenant_id': 'tenant-123',
        'plan': 'invite-only',
        'subscription_status': 'ACTIVE',
        'plan_code': 'core-pro',
        'licensed_gpu_count': 1,
        'trial_status': 'NONE',
    }
    
    result = check_compute_access(tenant)
    assert result is None  # Allowed


def test_check_compute_access_active_trial():
    """Test that active trial within window is allowed."""
    from handlers import check_compute_access
    
    # Trial ending in 7 days
    trial_end = datetime.now(timezone.utc) + timedelta(days=7)
    
    tenant = {
        'tenant_id': 'tenant-123',
        'plan': 'invite-only',
        'subscription_status': 'PENDING',
        'plan_code': 'core-pro',
        'licensed_gpu_count': 0,
        'trial_status': 'ACTIVE',
        'trial_end_at': trial_end.isoformat(),
        'trial_gpu_limit': 1,
    }
    
    result = check_compute_access(tenant)
    assert result is None  # Allowed


def test_check_compute_access_expired_trial():
    """Test that expired trial without subscription is blocked."""
    from handlers import check_compute_access
    
    # Trial ended 1 day ago
    trial_end = datetime.now(timezone.utc) - timedelta(days=1)
    
    tenant = {
        'tenant_id': 'tenant-123',
        'plan': 'invite-only',
        'subscription_status': 'PENDING',
        'plan_code': 'core-pro',
        'licensed_gpu_count': 0,
        'trial_status': 'EXPIRED',
        'trial_end_at': trial_end.isoformat(),
        'trial_gpu_limit': 0,
    }
    
    result = check_compute_access(tenant)
    assert result is not None  # Blocked
    assert result[0] == 402
    assert result[1]['error'] == 'subscription_required'


def test_check_compute_access_no_subscription_no_trial():
    """Test that tenant with no subscription and no trial is blocked."""
    from handlers import check_compute_access
    
    tenant = {
        'tenant_id': 'tenant-123',
        'plan': 'invite-only',
        'subscription_status': 'PENDING',
        'plan_code': None,
        'licensed_gpu_count': 0,
        'trial_status': 'NONE',
        'trial_end_at': None,
        'trial_gpu_limit': 0,
    }
    
    result = check_compute_access(tenant)
    assert result is not None  # Blocked
    assert result[0] == 402
    assert result[1]['error'] == 'subscription_required'


def test_check_compute_access_active_subscription_no_gpus():
    """Test that active subscription without licensed GPUs is blocked."""
    from handlers import check_compute_access
    
    tenant = {
        'tenant_id': 'tenant-123',
        'plan': 'invite-only',
        'subscription_status': 'ACTIVE',
        'plan_code': 'core-pro',
        'licensed_gpu_count': 0,  # No GPUs licensed
        'trial_status': 'NONE',
    }
    
    result = check_compute_access(tenant)
    assert result is not None  # Blocked
    assert result[0] == 402


def test_check_compute_access_cancelled_subscription():
    """Test that cancelled subscription is blocked."""
    from handlers import check_compute_access
    
    tenant = {
        'tenant_id': 'tenant-123',
        'plan': 'invite-only',
        'subscription_status': 'CANCELLED',
        'plan_code': 'core-pro',
        'licensed_gpu_count': 1,
        'trial_status': 'EXPIRED',
    }
    
    result = check_compute_access(tenant)
    assert result is not None  # Blocked
    assert result[0] == 402
