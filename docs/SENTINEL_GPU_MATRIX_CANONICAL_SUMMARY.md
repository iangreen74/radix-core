# Sentinel GPU Matrix + Agent Promotion Pipeline - Canonical Implementation

**Status**: ✅ CANONICAL - Ready for commit  
**Branch**: `feature/sentinel-gpu-matrix-promote-dev`  
**Date**: 2026-01-19

---

## 🎯 What Changed

This implementation makes the Sentinel GPU Matrix pipeline **canonical** by eliminating all "forgetting vectors" and enforcing deterministic, auditable agent promotion.

### A) Deterministic Non-Interactive Authentication

**Created**: `tools/sentinel/auth.py`

- Automatically obtains Cognito IdToken when `RADIX_JWT_TOKEN` not set
- Reads configuration from `ops/invariants.yaml` (single source of truth)
- Falls back to environment variables for CI compatibility
- No human interaction required for Sentinel contracts

**Modified**: `tools/sentinel/aws_gpu_matrix_runner.py`

- Removed manual JWT token passing
- Auto-authenticates via `auth.py` on initialization
- Simplified main() - only requires `RADIX_CANDIDATE_DIGEST`

### B) Pinned AMIs and Hardware Matrix

**Modified**: `ops/invariants.yaml`

- Added `agent_promotion` section to DEV environment
- Pinned NVIDIA AMI: `ami-0c2b8ca305e393e06` (Deep Learning AMI GPU PyTorch 2.1.0, Ubuntu 22.04, us-west-2)
- Pinned AMD AMI: `ami-0875d33dff2aae0d5` (Ubuntu 22.04 LTS, us-west-2)
- Encoded instance types, timeouts, cleanup TTL
- SSM parameter path and instance profile name

**Modified**: `tools/sentinel/aws_gpu_matrix_runner.py`

- Reads all configuration from invariants (no hardcoded values)
- `get_matrix_config()` function loads from YAML
- Instance provisioning uses config values

### C) Removed Fallback Sources for Approved Digest

**Modified**: `services/cloud-api/handlers.py` (`get_cluster_installer_handler`)

- **Removed** fallback to environment variable
- **Removed** hardcoded default digest
- SSM Parameter Store is **ONLY** source of truth
- Returns deterministic error codes on failure:
  - Exit 20: `RADIX_INSTALLER_ERROR: APPROVED_DIGEST_MISSING`
  - Exit 21: `RADIX_INSTALLER_ERROR: APPROVED_DIGEST_INVALID`
- Error messages include remediation steps

### D) Tightened Workflow Determinism

**Modified**: `.github/workflows/sentinel-gpu-matrix-and-promote-dev.yml`

1. **Concurrency Control**:
   ```yaml
   concurrency:
     group: sentinel-gpu-matrix-dev
     cancel-in-progress: false
   ```
   Prevents overlapping runs that could corrupt SSM parameter.

2. **Janitor Pre-Step**:
   - Terminates orphaned instances older than 4 hours
   - Runs before matrix tests
   - Uses `radix:purpose=gpu-matrix` tag filter

3. **Dry Run Mode**:
   - New input: `dry_run` (boolean)
   - Builds candidate image
   - Verifies manifest exists in GHCR
   - Skips EC2 provisioning
   - Useful for testing build process

4. **Simplified Auth**:
   - Removed manual JWT token generation
   - Uses `auth.py` module
   - Only requires `RADIX_TEST_USERNAME` and `RADIX_TEST_PASSWORD`

5. **Evidence Pack Upload**:
   - Already had `retention-days: 90`
   - Now uploads even on failure (`if: always()`)

### E) Enhanced Evidence Pack

**Modified**: `tools/sentinel/reporting/evidence_pack.py`

- Added `approved_digest_before` field (captured before tests)
- Added `approved_digest_after` field (reflects promotion result)
- Added `time_to_active_seconds` metric (onboarding → cluster active)
- Added `remediation` field for failures
- Enhanced Markdown report with before/after digest comparison

**Modified**: `tools/sentinel/aws_gpu_matrix_runner.py`

- Captures approved digest before tests start
- Passes to EvidencePack constructor
- Tracks time-to-active for each vendor test
- Includes remediation hints in failure results

---

## 📦 Files Modified (10 files)

### New Files (2)

1. **`tools/sentinel/auth.py`** (+180 lines)
   - Cognito authentication helper
   - Reads from invariants.yaml
   - Deterministic, non-interactive

2. **`tools/sentinel/reporting/__init__.py`** (+5 lines)
   - Package initialization

### Modified Files (8)

3. **`tools/sentinel/aws_gpu_matrix_runner.py`** (major refactor)
   - Auto-authentication via `auth.py`
   - Configuration from invariants.yaml
   - Enhanced evidence pack with before/after digests
   - Time-to-active tracking
   - Remediation hints

4. **`tools/sentinel/reporting/evidence_pack.py`** (enhanced)
   - `approved_digest_before` and `approved_digest_after` fields
   - `time_to_active_seconds` metric
   - `remediation` field for failures
   - Enhanced Markdown formatting

5. **`tools/sentinel/aws_ssm.py`** (no changes needed - already canonical)

6. **`.github/workflows/sentinel-gpu-matrix-and-promote-dev.yml`** (hardened)
   - Concurrency control
   - Janitor pre-step
   - Dry run mode
   - Simplified auth (uses auth.py)
   - pyyaml dependency added

7. **`services/cloud-api/handlers.py`** (hardened)
   - Removed fallback digest sources
   - SSM-only source of truth
   - Deterministic error codes (20, 21)
   - Actionable error messages

8. **`infra/cloud/template.yaml`** (no additional changes - SSM policy already added)

9. **`ops/invariants.yaml`** (enhanced)
   - `agent_promotion` section with full config
   - Pinned AMI IDs
   - Instance types, timeouts, cleanup settings

10. **`docs/canonical/SENTINEL_CONTRACTS_AND_INVARIANTS.md`** (updated)
    - GPU Matrix pipeline documentation
    - Hardware matrix details
    - Determinism guarantees section

---

## 🛡️ Forgetting Vectors Eliminated

| Vector | Before | After |
|--------|--------|-------|
| **Auth** | Manual JWT token required | Auto-auth via Cognito |
| **AMIs** | Hardcoded, could drift | Pinned in invariants.yaml |
| **Approved Digest** | Fallback to env var | SSM only, deterministic errors |
| **Concurrency** | Multiple runs could overlap | Concurrency group enforces serial |
| **Orphaned Instances** | Manual cleanup required | Janitor auto-cleanup |
| **Config Drift** | Hardcoded in Python | Single source: invariants.yaml |
| **Evidence on Failure** | May not upload | Always uploads (if: always()) |
| **Time Metrics** | Only onboarding duration | Added time-to-active |
| **Remediation** | Generic error messages | Specific remediation hints |

---

## 🔐 Determinism Guarantees

1. **No Human Interaction**: Auth happens automatically via Cognito
2. **No Mutable References**: AMIs pinned by ID, digest by SHA256
3. **Single Source of Truth**: SSM for approved digest, invariants.yaml for config
4. **Serial Execution**: Concurrency control prevents race conditions
5. **Cost Safety**: Janitor cleanup, TTL tags, timeouts on all operations
6. **Audit Trail**: Evidence pack always uploaded, includes before/after state
7. **Reproducible**: Same git SHA + same AMIs = same test environment

---

## 📋 Manual Git Commit Instructions

```bash
# Ensure you're on the correct branch
git checkout -b feature/sentinel-gpu-matrix-promote-dev

# Stage all modified files
git add tools/sentinel/auth.py
git add tools/sentinel/reporting/__init__.py
git add tools/sentinel/aws_gpu_matrix_runner.py
git add tools/sentinel/reporting/evidence_pack.py
git add tools/sentinel/aws_ssm.py
git add .github/workflows/sentinel-gpu-matrix-and-promote-dev.yml
git add services/cloud-api/handlers.py
git add infra/cloud/template.yaml
git add ops/invariants.yaml
git add docs/canonical/SENTINEL_CONTRACTS_AND_INVARIANTS.md

# Commit with detailed message
git commit -m "feat: Sentinel GPU Matrix + Agent Promotion (canonical)

Implements deterministic, auditable agent promotion pipeline that tests
candidate images on real NVIDIA + AMD GPU hardware before promotion.

ELIMINATES FORGETTING VECTORS:
- Auto-auth via Cognito (no manual JWT)
- Pinned AMIs in invariants.yaml (no drift)
- SSM-only approved digest (no fallbacks)
- Concurrency control (no race conditions)
- Janitor cleanup (no orphaned instances)

NEW COMPONENTS:
- tools/sentinel/auth.py: Deterministic Cognito authentication
- tools/sentinel/aws_gpu_matrix_runner.py: GPU matrix test harness
- tools/sentinel/aws_ssm.py: SSM helper utilities
- tools/sentinel/reporting/evidence_pack.py: Evidence generation
- .github/workflows/sentinel-gpu-matrix-and-promote-dev.yml: CI workflow

HARDENED COMPONENTS:
- services/cloud-api/handlers.py: SSM-only digest source, deterministic errors
- ops/invariants.yaml: Pinned AMIs, full agent_promotion config
- docs/canonical/SENTINEL_CONTRACTS_AND_INVARIANTS.md: Pipeline docs

WORKFLOW FEATURES:
- Builds candidate agent image → GHCR
- Provisions NVIDIA (g5.xlarge) + AMD (g4ad.xlarge) instances
- Runs control-plane installer end-to-end
- Validates GPU detection, runtime, onboarding, job execution
- Generates evidence pack (JSON + Markdown, 90 day retention)
- Promotes digest to SSM if both tests pass
- Janitor cleanup for orphaned instances
- Dry run mode for build-only testing
- Concurrency control for serial execution

EVIDENCE PACK INCLUDES:
- Git SHA, run ID, timestamps
- Candidate digest, approved digest before/after
- Instance metadata (ID, AMI, type, region)
- GPU info (vendor, model, driver version)
- Onboarding metrics (duration, time-to-active)
- Smoke job results (status, duration)
- Failure codes and remediation hints

DETERMINISM:
- No human-supplied credentials
- No mutable image references
- No config drift
- No race conditions
- Always auditable

This is the permanent 'no forgetting' mechanism for agent releases."

# Push to remote
git push origin feature/sentinel-gpu-matrix-promote-dev

# Open PR via GitHub UI or gh CLI
gh pr create \
  --title "feat: Sentinel GPU Matrix + Agent Promotion (canonical)" \
  --body "See commit message for full details. This PR implements the permanent 'no forgetting' mechanism for agent releases."
```

---

## 🚀 Next Steps After Merge

1. **Create IAM Instance Profile**:
   ```bash
   aws iam create-role \
     --role-name RadixSentinelSSMProfile \
     --assume-role-policy-document file://sentinel-trust-policy.json
   
   aws iam attach-role-policy \
     --role-name RadixSentinelSSMProfile \
     --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
   
   aws iam create-instance-profile \
     --instance-profile-name RadixSentinelSSMProfile
   
   aws iam add-role-to-instance-profile \
     --instance-profile-name RadixSentinelSSMProfile \
     --role-name RadixSentinelSSMProfile
   ```

2. **Initialize SSM Parameter** (bootstrap):
   ```bash
   # Set initial approved digest (use current production digest)
   aws ssm put-parameter \
     --name /radix/dev/agent_digest_approved \
     --value "ghcr.io/iangreen74/radix-cluster-agent@sha256:CURRENT_DIGEST" \
     --type String \
     --description "Approved agent digest promoted by Sentinel GPU matrix pipeline"
   ```

3. **Test Workflow Manually**:
   - Go to Actions → "Sentinel GPU Matrix + Agent Promotion (DEV)"
   - Click "Run workflow"
   - Select: `dry_run: true` (first test)
   - Verify candidate image builds and manifest check passes
   - Run again with `dry_run: false` to test full matrix

4. **Monitor First Full Run**:
   - Check evidence pack artifacts
   - Verify NVIDIA and AMD tests pass
   - Confirm digest promotion to SSM
   - Review CloudWatch logs for any issues

5. **Enable Nightly Schedule**:
   - Already configured: 2am PT daily
   - Will run automatically after merge

---

## 🎓 How This Prevents Regressions

Every agent image must **prove itself on real GPU hardware** before users see it:

1. **GPU Detection**: Tested on actual NVIDIA and AMD GPUs
2. **Container Runtime**: nvidia-container-toolkit and ROCm validated
3. **Installer Script**: Control-plane generated installer tested end-to-end
4. **Onboarding Flow**: Token exchange and cluster activation verified
5. **GPU Jobs**: Smoke job proves agent can schedule GPU workloads
6. **Deterministic**: Same SHA + same AMIs = same test environment
7. **Auditable**: Evidence pack documents everything
8. **Gated**: Both NVIDIA AND AMD must pass for promotion

**The system cannot "forget" that an agent works** because SSM parameter is the single source of truth, and it's only updated by passing GPU tests.

---

## ✅ Definition of Done

- ✅ Deterministic auth implemented (auth.py)
- ✅ AMIs pinned in invariants.yaml
- ✅ Approved digest source hardened (SSM-only)
- ✅ Workflow concurrency control added
- ✅ Janitor cleanup implemented
- ✅ Dry run mode added
- ✅ Evidence pack enhanced (before/after, time-to-active, remediation)
- ✅ Documentation updated
- ⏸️ Manual commit/push required
- ⏸️ IAM instance profile creation required
- ⏸️ SSM parameter initialization required
- ⏸️ First workflow run required

---

**This implementation is canonical and merge-ready.**
