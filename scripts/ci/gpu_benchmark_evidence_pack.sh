#!/usr/bin/env bash
#
# GPU Benchmark Evidence Pack - Sequential Trial Runner
#
# Runs multiple benchmark trials sequentially with:
# - AWS credential refresh per trial
# - Cluster creation + token management
# - Terraform infrastructure provisioning
# - Job launch and polling
# - Observation generation
# - Cleanup with credential refresh
#
# Required environment variables:
#   TRIALS, INSTANCE_TYPE, BATCH_SIZE, EPOCHS, AWS_REGION, RETAIN_ON_FAILURE
#   GITHUB_RUN_ID, VPC_ID, SUBNET_ID, AMI_ID
#   RADIX_ID_TOKEN
#
# AWS credentials should be pre-configured by workflow (configure-aws-credentials action)

set -euo pipefail

# Resolve repository root directory
ROOT_DIR="${GITHUB_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RADIX_API="${ROOT_DIR}/scripts/ci/radix_api.sh"
OBS_DIR="${ROOT_DIR}/observations"

# Create observations directory
mkdir -p "$OBS_DIR"

# Verify radix_api.sh exists
if [ ! -x "$RADIX_API" ]; then
  echo "❌ ERROR: radix_api.sh not found or not executable at $RADIX_API" >&2
  exit 1
fi

# Logging helpers
log_info() { echo "ℹ️  $*"; }
log_success() { echo "✓ $*"; }
log_error() { echo "❌ $*" >&2; }
log_warn() { echo "⚠️  $*"; }

# Bootstrap invariant checks - prevent regression of GHCR auth logic
log_info "Checking bootstrap template for forbidden patterns..."
BOOTSTRAP_TEMPLATE="${ROOT_DIR}/infra/bench_harness/user_data.sh.tftpl"

if grep -q "/radix/github/ghcr_token" "$BOOTSTRAP_TEMPLATE"; then
  log_error "BOOTSTRAP INVARIANT VIOLATION: user_data.sh.tftpl contains forbidden '/radix/github/ghcr_token'"
  log_error "The radix-cluster-agent image is PUBLIC and does not require authentication."
  log_error "Remove all GHCR_TOKEN references from the bootstrap template."
  exit 1
fi

if grep -q "docker login ghcr.io" "$BOOTSTRAP_TEMPLATE"; then
  log_error "BOOTSTRAP INVARIANT VIOLATION: user_data.sh.tftpl contains forbidden 'docker login ghcr.io'"
  log_error "The radix-cluster-agent image is PUBLIC and does not require authentication."
  log_error "Remove all docker login commands from the bootstrap template."
  exit 1
fi

log_success "Bootstrap template passes invariant checks"

# API base guardrail - prevent running against PROD
log_info "Checking API base configuration..."
if [ "${RADIX_API_BASE:-}" = "https://api.vaultscaler.com" ]; then
  log_error "API BASE GUARDRAIL VIOLATION: Refusing to run Evidence Pack against PROD API"
  log_error "Current RADIX_API_BASE: ${RADIX_API_BASE}"
  log_error "Evidence Pack benchmarks must use DEV API base (e.g., https://api.dev.vaultscaler.com)"
  log_error "Update the workflow api_base input to use the correct DEV environment."
  exit 1
fi

if [ -z "${RADIX_API_BASE:-}" ]; then
  log_error "RADIX_API_BASE environment variable is not set"
  log_error "This must be set by the workflow to the DEV API base URL"
  exit 1
fi

log_success "API base check passed: ${RADIX_API_BASE}"

# DNS preflight check
log_info "Checking DNS resolution for API base..."
API_HOSTNAME=$(echo "${RADIX_API_BASE}" | sed -E 's|^https?://([^/]+).*|\1|')

if ! python3 -c "import socket; socket.gethostbyname('${API_HOSTNAME}')" 2>/dev/null; then
  log_error "DNS PREFLIGHT FAILURE: Cannot resolve ${API_HOSTNAME}"
  log_error "RADIX_API_BASE: ${RADIX_API_BASE}"
  log_error ""
  log_error "This hostname does not resolve. Possible causes:"
  log_error "  1. Custom domain not configured in API Gateway"
  log_error "  2. Route53 record missing or incorrect"
  log_error "  3. Wrong environment (use execute-api invoke URL for DEV)"
  log_error ""
  log_error "To find the correct DEV invoke URL:"
  log_error "  aws cloudformation describe-stacks --stack-name <DEV_STACK_NAME> --query 'Stacks[0].Outputs[?OutputKey==\`ApiBaseUrl\`].OutputValue' --output text"
  exit 1
fi

log_success "DNS resolution successful: ${API_HOSTNAME}"

# Print authentication context for debugging
log_info "Authentication context:"
log_info "  RADIX_API_BASE: ${RADIX_API_BASE}"
if [ -n "${RADIX_ID_TOKEN:-}" ]; then
  log_info "  RADIX_ID_TOKEN: set (length: ${#RADIX_ID_TOKEN})"
else
  log_error "  RADIX_ID_TOKEN: NOT SET"
  exit 1
fi

# Validate required environment variables
: "${TRIALS:?TRIALS not set}"
: "${INSTANCE_TYPE:?INSTANCE_TYPE not set}"
: "${BATCH_SIZE:?BATCH_SIZE not set}"
: "${EPOCHS:?EPOCHS not set}"
: "${AWS_REGION:?AWS_REGION not set}"
: "${RETAIN_ON_FAILURE:?RETAIN_ON_FAILURE not set}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID not set}"
: "${VPC_ID:?VPC_ID not set}"
: "${SUBNET_ID:?SUBNET_ID not set}"
: "${AMI_ID:?AMI_ID not set}"
: "${RADIX_ID_TOKEN:?RADIX_ID_TOKEN not set}"

# Global trial state
SUCCESSFUL_TRIALS=0
FAILED_TRIALS=0

# Cleanup trial resources
cleanup_trial() {
  local vpc_id="$1"
  local subnet_id="$2"
  local ami_id="$3"
  local instance_type="$4"
  local region="$5"
  local run_id="$6"
  local tenant_id="$7"
  local cluster_id="$8"
  local ssm_param_name="$9"
  
  log_info "Cleaning up trial resources..."
  
  if [ -d "infra/bench_harness" ]; then
    cd infra/bench_harness
    
    terraform destroy -input=false -auto-approve \
      -var="vpc_id=${vpc_id}" \
      -var="subnet_id=${subnet_id}" \
      -var="ami_id=${ami_id}" \
      -var="instance_type=${instance_type}" \
      -var="region=${region}" \
      -var="run_id=${run_id}" \
      -var="radix_api_base=${RADIX_API_BASE}" \
      -var="tenant_id=${tenant_id}" \
      -var="cluster_id=${cluster_id}" \
      -var="ssm_token_param=${ssm_param_name}" || true
    
    cd ../..
  fi
  
  # Cleanup SSM parameter
  aws ssm delete-parameter --name "${ssm_param_name}" --region "${region}" 2>/dev/null || true
  
  log_success "Trial cleanup completed"
}

# Run a single benchmark trial
run_trial() {
  local trial_num="$1"
  
  echo "=========================================="
  echo "Trial ${trial_num} of ${TRIALS}"
  echo "=========================================="
  
  # Generate identifiers
  local run_id="${GITHUB_RUN_ID}-t${trial_num}"
  local trial_id="trial_${trial_num}"
  
  # Generate unique cluster name: ci-gpu-bench-{run_id}-b{batch}-t{trial}
  # Sanitize run_id to lowercase alphanumeric + hyphens, limit length to avoid K8s 63-char limit
  local sanitized_run_id=$(echo "${GITHUB_RUN_ID}" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]-' | cut -c1-20)
  local cluster_name="ci-gpu-b${BATCH_SIZE}-${sanitized_run_id}-t${trial_num}"
  
  log_info "Trial ID: ${trial_id}"
  log_info "Run ID: ${run_id}"
  log_info "Cluster Name: ${cluster_name}"
  
  # Create diagnostics directory early for failure capture
  local diag_dir="${ROOT_DIR}/artifacts/diagnostics_trial_${trial_num}"
  mkdir -p "${diag_dir}"
  
  # Capture Cognito token debug info (from get_ci_token.py stderr)
  log_info "Capturing Cognito token debug info..."
  python3 "${ROOT_DIR}/scripts/get_ci_token.py" > /dev/null 2>"${diag_dir}/cognito_token_debug.txt" || true
  
  # Validate token issuer matches API base environment
  log_info "Validating token issuer vs API base..."
  TOKEN_ISSUER=$(grep "Token Issuer" "${diag_dir}/cognito_token_debug.txt" | sed 's/.*: //' || echo "")
  
  if [ -n "$TOKEN_ISSUER" ]; then
    # Extract region from API base (e.g., us-west-2 from https://abc.execute-api.us-west-2.amazonaws.com/dev)
    API_REGION=$(echo "${RADIX_API_BASE}" | grep -oP '(?<=execute-api\.)[^.]+' || echo "")
    
    # Extract region from token issuer (e.g., us-west-2 from https://cognito-idp.us-west-2.amazonaws.com/us-west-2_abc123)
    TOKEN_REGION=$(echo "$TOKEN_ISSUER" | grep -oP '(?<=cognito-idp\.)[^.]+' || echo "")
    
    if [ -n "$API_REGION" ] && [ -n "$TOKEN_REGION" ] && [ "$API_REGION" != "$TOKEN_REGION" ]; then
      log_error "TOKEN/API REGION MISMATCH DETECTED"
      log_error "  Token Issuer Region: $TOKEN_REGION"
      log_error "  API Base Region: $API_REGION"
      log_error "  RADIX_API_BASE: ${RADIX_API_BASE}"
      log_error ""
      log_error "The Cognito token is from a different region than the API Gateway."
      log_error "This will cause authentication failures."
      log_error ""
      log_error "Fix: Ensure RADIX_COGNITO_USER_POOL_ID matches the DEV environment region."
      log_error "See diagnostics: ${diag_dir}/cognito_token_debug.txt"
      return 1
    fi
    
    log_success "Token issuer validation passed (region: $TOKEN_REGION)"
  else
    log_warn "Could not extract token issuer for validation"
  fi
  
  # Create Radix cluster
  log_info "Creating cluster via Radix API..."
  local cluster_resp
  local create_cluster_exit_code=0
  
  # Set debug output path for API errors
  export RADIX_API_DEBUG_OUT="${diag_dir}/create_cluster_api_response.txt"
  
  # Capture stdout and stderr separately
  cluster_resp=$("$RADIX_API" create-cluster "${cluster_name}" "docker_host" 2>"${diag_dir}/create_cluster_api_stderr.txt") || create_cluster_exit_code=$?
  echo "${cluster_resp}" > "${diag_dir}/create_cluster_api_stdout.txt"
  
  local cluster_id
  cluster_id=$(echo "${cluster_resp}" | jq -r '.cluster_id // .id // empty')
  
  local tenant_id
  tenant_id=$(echo "${cluster_resp}" | jq -r '.tenant_id // empty')
  
  if [ -z "${cluster_id}" ] || [ ${create_cluster_exit_code} -ne 0 ]; then
    log_error "Failed to create cluster"
    
    # Write comprehensive failure diagnostics
    {
      echo "Create Cluster Failure Diagnostics"
      echo "==================================="
      echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "Trial: ${trial_num}"
      echo "Cluster Name: ${cluster_name}"
      echo ""
      echo "Environment:"
      echo "  RADIX_API_BASE: ${RADIX_API_BASE}"
      echo "  RADIX_ID_TOKEN: set (length: ${#RADIX_ID_TOKEN})"
      echo ""
      echo "Exit Code: ${create_cluster_exit_code}"
    } > "${diag_dir}/create_cluster_failure.txt"
    
    # Decode actual RADIX_ID_TOKEN used in the failed request
    log_info "Decoding RADIX_ID_TOKEN claims..."
    if [ -n "${RADIX_ID_TOKEN}" ]; then
      python3 -c "
import json
import base64
import sys
from datetime import datetime

token = '${RADIX_ID_TOKEN}'
try:
    parts = token.split('.')
    if len(parts) != 3:
        sys.exit(1)
    
    # Decode payload (add padding if needed)
    payload = parts[1]
    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += '=' * padding
    
    decoded = base64.urlsafe_b64decode(payload)
    claims = json.loads(decoded)
    
    # Write full claims JSON
    with open('${diag_dir}/radix_id_token_claims.json', 'w') as f:
        json.dump(claims, f, indent=2)
    
    # Write human-readable summary
    with open('${diag_dir}/radix_id_token_summary.txt', 'w') as f:
        f.write('RADIX ID Token Summary\\n')
        f.write('======================\\n\\n')
        f.write(f\"iss: {claims.get('iss', 'N/A')}\\n\")
        f.write(f\"aud: {claims.get('aud', 'N/A')}\\n\")
        f.write(f\"token_use: {claims.get('token_use', 'N/A')}\\n\")
        
        exp = claims.get('exp', 0)
        exp_human = datetime.utcfromtimestamp(exp).strftime('%Y-%m-%d %H:%M:%S UTC') if exp else 'N/A'
        f.write(f\"exp: {exp} ({exp_human})\\n\")
        
        username = claims.get('cognito:username') or claims.get('username', 'N/A')
        f.write(f\"cognito:username: {username}\\n\")
        
        if 'client_id' in claims:
            f.write(f\"client_id: {claims.get('client_id')}\\n\")
        
        f.write(f\"\\nAPI base called: ${RADIX_API_BASE}\\n\")
    
    print('Token claims decoded successfully', file=sys.stderr)
except Exception as e:
    print(f'Failed to decode token: {e}', file=sys.stderr)
    sys.exit(1)
" || log_warn "Failed to decode RADIX_ID_TOKEN"
    else
      log_warn "RADIX_ID_TOKEN not set, cannot decode"
    fi
    
    # Test API connectivity
    log_info "Testing API connectivity..."
    {
      echo "=== API Health Check ==="
      curl -Is "${RADIX_API_BASE}/health" 2>&1 | head -n 20 || echo "Health endpoint failed"
      echo ""
      echo "=== API Root Check ==="
      curl -Is "${RADIX_API_BASE}/" 2>&1 | head -n 20 || echo "Root endpoint failed"
    } > "${diag_dir}/api_connectivity.txt"
    
    log_error "Diagnostics saved to: ${diag_dir}"
    return 1
  fi
  
  [ -n "${cluster_id}" ] && echo "::add-mask::${cluster_id}"
  [ -n "${tenant_id}" ] && echo "::add-mask::${tenant_id}"
  log_success "Cluster created: ${cluster_id:0:20}..."
  
  # Get tenant ID if not in response
  if [ -z "${tenant_id}" ]; then
    tenant_id=$("$RADIX_API" get-tenant-id | jq -r '.')
    [ -n "${tenant_id}" ] && echo "::add-mask::${tenant_id}"
  fi
  
  # Create onboarding token
  log_info "Creating onboarding token..."
  local ttl_hours=1
  local token_resp
  token_resp=$("$RADIX_API" create-token "${cluster_id}" "${ttl_hours}")
  
  local one_time_token
  one_time_token=$(echo "${token_resp}" | jq -r '.one_time_token')
  
  if [ -z "${one_time_token}" ]; then
    log_error "Failed to create onboarding token"
    return 1
  fi
  
  [ -n "${one_time_token}" ] && echo "::add-mask::${one_time_token}"
  log_success "Onboarding token created"
  
  # Store token in SSM
  local ssm_param_name="/radix/bench/evidence/${GITHUB_RUN_ID}/trial_${trial_num}/token"
  aws ssm put-parameter \
    --name "${ssm_param_name}" \
    --value "${one_time_token}" \
    --type SecureString \
    --overwrite \
    --region "${AWS_REGION}"
  log_success "Token stored in SSM: ${ssm_param_name}"
  
  # Run benchmark harness
  cd infra/bench_harness
  
  # Initialize Terraform
  terraform init -input=false
  
  # Apply infrastructure
  log_info "Creating EC2 instance..."
  terraform apply -input=false -auto-approve \
    -var="vpc_id=${VPC_ID}" \
    -var="subnet_id=${SUBNET_ID}" \
    -var="ami_id=${AMI_ID}" \
    -var="instance_type=${INSTANCE_TYPE}" \
    -var="region=${AWS_REGION}" \
    -var="run_id=${run_id}" \
    -var="radix_api_base=${RADIX_API_BASE}" \
    -var="tenant_id=${tenant_id}" \
    -var="cluster_id=${cluster_id}" \
    -var="ssm_token_param=${ssm_param_name}" \
    -var="agent_image=ghcr.io/iangreen74/radix-cluster-agent:latest" \
    -var="retain=${RETAIN_ON_FAILURE}"
  
  # Get instance details
  local instance_id
  instance_id=$(terraform output -raw bench_instance_id 2>/dev/null || echo "")
  
  if [ -z "${instance_id}" ]; then
    log_error "Failed to get instance ID"
    cleanup_trial "${VPC_ID}" "${SUBNET_ID}" "${AMI_ID}" "${INSTANCE_TYPE}" "${AWS_REGION}" "${run_id}" "${tenant_id}" "${cluster_id}" "${ssm_param_name}"
    cd ../..
    return 1
  fi
  
  log_info "Instance ID: ${instance_id}"
  
  # Wait for cluster to become active
  log_info "Waiting for cluster to become active..."
  local max_attempts=60
  local attempt=0
  local cluster_active=false
  
  while [ ${attempt} -lt ${max_attempts} ]; do
    attempt=$((attempt + 1))
    
    local cluster_status_resp
    cluster_status_resp=$("$RADIX_API" get-cluster "${cluster_id}")
    
    local cluster_status
    cluster_status=$(echo "${cluster_status_resp}" | jq -r '.status // "unknown"')
    
    log_info "Attempt ${attempt}/${max_attempts}: Cluster status: ${cluster_status}"
    
    if [ "${cluster_status}" = "active" ]; then
      log_success "Cluster is active"
      cluster_active=true
      break
    fi
    
    sleep 10
  done
  
  if [ "${cluster_active}" != "true" ]; then
    log_error "Cluster did not become active within timeout"
    
    # DNS preflight check
    log_info "Checking DNS resolution for API base..."
    API_HOSTNAME=$(echo "${RADIX_API_BASE}" | sed -E 's|^https?://([^/]+).*|\1|')

    if ! python3 -c "import socket; socket.gethostbyname('${API_HOSTNAME}')" 2>/dev/null; then
      log_error "DNS PREFLIGHT FAILURE: Cannot resolve ${API_HOSTNAME}"
      log_error "RADIX_API_BASE: ${RADIX_API_BASE}"
      log_error ""
      log_error "This hostname does not resolve. Possible causes:"
      log_error "  1. Custom domain not configured in API Gateway"
      log_error "  2. Route53 record missing or incorrect"
      log_error "  3. Wrong environment (use execute-api invoke URL for DEV)"
      log_error ""
      log_error "To find the correct DEV invoke URL:"
      log_error "  aws cloudformation describe-stacks --stack-name <DEV_STACK_NAME> --query 'Stacks[0].Outputs[?OutputKey==\`ApiBaseUrl\`].OutputValue' --output text"
      exit 1
    fi

    log_success "DNS resolution successful: ${API_HOSTNAME}"

    log_success "API base check passed: ${RADIX_API_BASE}"

    # DIAGNOSTIC CAPTURE: Save failure context for debugging
    local diag_dir="${ROOT_DIR}/artifacts/diagnostics_trial_${trial_num}"
    mkdir -p "${diag_dir}"
    
    log_info "Capturing diagnostic information..."
    
    # 1. Terraform outputs
    terraform output -json > "${diag_dir}/terraform_outputs.json" 2>&1 || echo "{}" > "${diag_dir}/terraform_outputs.json"
    
    # 2. EC2 instance details
    if [ -n "${instance_id}" ]; then
      aws ec2 describe-instances --instance-ids "${instance_id}" --region "${AWS_REGION}" \
        > "${diag_dir}/ec2_describe_instance.json" 2>&1 || echo "Failed to describe instance" > "${diag_dir}/ec2_describe_instance.json"
      
      aws ec2 describe-instance-status --instance-ids "${instance_id}" --region "${AWS_REGION}" \
        > "${diag_dir}/ec2_instance_status.json" 2>&1 || echo "Failed to get instance status" > "${diag_dir}/ec2_instance_status.json"
    fi
    
    # 3. SSM parameter verification
    aws ssm get-parameter --name "${ssm_param_name}" --region "${AWS_REGION}" --with-decryption \
      > "${diag_dir}/ssm_parameter_check.json" 2>&1 || echo "Failed to retrieve SSM parameter" > "${diag_dir}/ssm_parameter_check.json"
    
    # 4. Cluster API response (last status check)
    if [ -n "${cluster_status_resp}" ]; then
      echo "${cluster_status_resp}" | jq '.' > "${diag_dir}/cluster_status_final.json" 2>&1 || echo "${cluster_status_resp}" > "${diag_dir}/cluster_status_final.json"
    fi
    
    # 5. SSM command history (check if agent bootstrap ran)
    if [ -n "${instance_id}" ]; then
      aws ssm list-commands --instance-id "${instance_id}" --region "${AWS_REGION}" --max-results 10 \
        > "${diag_dir}/ssm_command_history.json" 2>&1 || echo "Failed to list SSM commands" > "${diag_dir}/ssm_command_history.json"
    fi
    
    # 6. CloudWatch Logs (if agent logs exist)
    local log_group="/radix/cluster-agent/${cluster_id}"
    aws logs describe-log-streams --log-group-name "${log_group}" --region "${AWS_REGION}" --max-items 5 \
      > "${diag_dir}/cloudwatch_log_streams.json" 2>&1 || echo "No CloudWatch logs found" > "${diag_dir}/cloudwatch_log_streams.json"
    
    # 7. Fetch logs from instance via SSM (automated debugging)
    log_info "Fetching logs from instance via SSM..."
    
    # Determine instance_id if not already set
    local fetch_instance_id="${instance_id}"
    if [ -z "${fetch_instance_id}" ]; then
      fetch_instance_id=$(terraform output -json 2>/dev/null | jq -r '.bench_instance_id.value // empty' || echo "")
    fi
    
    if [ -n "${fetch_instance_id}" ]; then
      # Send SSM command to fetch logs
      local ssm_command_id
      ssm_command_id=$(aws ssm send-command \
        --instance-ids "${fetch_instance_id}" \
        --document-name "AWS-RunShellScript" \
        --parameters 'commands=[
          "echo \"=== Bootstrap Log ===\"",
          "sudo tail -n 200 /var/log/radix-bootstrap.log 2>&1 || echo \"bootstrap log not found\"",
          "echo \"\"",
          "echo \"=== Agent Log ===\"",
          "sudo tail -n 200 /var/log/radix-agent.log 2>&1 || echo \"agent log not found\"",
          "echo \"\"",
          "echo \"=== Docker Containers ===\"",
          "sudo docker ps -a 2>&1 || echo \"docker ps failed\"",
          "echo \"\"",
          "echo \"=== Docker Agent Logs ===\"",
          "sudo docker logs --tail 200 radix-agent 2>&1 || echo \"docker logs failed\"",
          "echo \"\"",
          "echo \"=== Network Check: Radix API ===\"",
          "curl -Is https://api.vaultscaler.com 2>&1 | head -n 5 || echo \"API check failed\"",
          "echo \"\"",
          "echo \"=== Network Check: GHCR ===\"",
          "curl -Is https://ghcr.io 2>&1 | head -n 5 || echo \"GHCR check failed\""
        ]' \
        --region "${AWS_REGION}" \
        --output text \
        --query 'Command.CommandId' 2>&1)
      
      if [ -n "${ssm_command_id}" ] && [ "${ssm_command_id}" != "None" ]; then
        log_info "SSM command sent: ${ssm_command_id}"
        
        # Wait for command to complete (max 60 seconds)
        local wait_count=0
        local command_status="Pending"
        while [ ${wait_count} -lt 12 ] && [ "${command_status}" != "Success" ] && [ "${command_status}" != "Failed" ] && [ "${command_status}" != "TimedOut" ]; do
          sleep 5
          command_status=$(aws ssm get-command-invocation \
            --command-id "${ssm_command_id}" \
            --instance-id "${fetch_instance_id}" \
            --region "${AWS_REGION}" \
            --query 'Status' \
            --output text 2>&1 || echo "Failed")
          wait_count=$((wait_count + 1))
        done
        
        log_info "SSM command status: ${command_status}"
        
        # Retrieve command output
        if [ "${command_status}" = "Success" ] || [ "${command_status}" = "Failed" ]; then
          local ssm_output
          ssm_output=$(aws ssm get-command-invocation \
            --command-id "${ssm_command_id}" \
            --instance-id "${fetch_instance_id}" \
            --region "${AWS_REGION}" \
            --query 'StandardOutputContent' \
            --output text 2>&1)
          
          if [ -n "${ssm_output}" ]; then
            # Redact sensitive tokens before saving
            echo "${ssm_output}" | sed 's/RADIX_ONE_TIME_TOKEN=[^ ]*/RADIX_ONE_TIME_TOKEN=***REDACTED***/g' \
              | sed 's/"token":"[^"]*"/"token":"***REDACTED***"/g' \
              > "${diag_dir}/instance_logs_via_ssm.txt"
            log_success "Instance logs fetched via SSM"
          else
            echo "SSM command completed but no output received" > "${diag_dir}/ssm_fetch_error.txt"
          fi
        else
          echo "SSM command failed or timed out" > "${diag_dir}/ssm_fetch_error.txt"
          echo "Command ID: ${ssm_command_id}" >> "${diag_dir}/ssm_fetch_error.txt"
          echo "Status: ${command_status}" >> "${diag_dir}/ssm_fetch_error.txt"
        fi
      else
        echo "Failed to send SSM command" > "${diag_dir}/ssm_fetch_error.txt"
        echo "Command output: ${ssm_command_id}" >> "${diag_dir}/ssm_fetch_error.txt"
      fi
    else
      echo "Instance ID not available for SSM fetch" > "${diag_dir}/ssm_fetch_error.txt"
    fi
    
    # 8. Summary of diagnostic findings
    cat > "${diag_dir}/DIAGNOSTIC_SUMMARY.txt" <<DIAGEOF
Cluster Activation Failure Diagnostics
======================================
Trial: ${trial_num}
Cluster ID: ${cluster_id}
Instance ID: ${instance_id}
Run ID: ${run_id}
SSM Parameter: ${ssm_param_name}

Failure: Cluster did not transition to 'active' status within 10 minutes (60 attempts)

Likely causes:
1. EC2 instance failed to reach running state (check ec2_describe_instance.json)
2. Cluster-agent container failed to start (check SSM command history)
3. Agent could not retrieve onboarding token from SSM (check ssm_parameter_check.json)
4. Network connectivity issue (security group, no internet gateway)
5. Agent failed to register with Radix API (check cluster_status_final.json)

Next steps:
- Review ec2_describe_instance.json for instance state and status checks
- Check ssm_command_history.json to see if user-data script ran
- Verify ssm_parameter_check.json shows token was stored correctly
- If instance is running, SSH in (if retain_on_failure=true) and check:
  - docker ps (is cluster-agent container running?)
  - docker logs <container-id>
  - journalctl -u cloud-init-output.log
DIAGEOF
    
    log_success "Diagnostics saved to: ${diag_dir}"
    
    # Cleanup trial resources (unless retain_on_failure is true)
    if [ "${RETAIN_ON_FAILURE}" != "true" ]; then
      cleanup_trial "${VPC_ID}" "${SUBNET_ID}" "${AMI_ID}" "${INSTANCE_TYPE}" "${AWS_REGION}" "${run_id}" "${tenant_id}" "${cluster_id}" "${ssm_param_name}"
    else
      log_warn "Instance retained for debugging (retain_on_failure=true)"
      log_info "Instance ID: ${instance_id}"
      log_info "Cluster ID: ${cluster_id}"
      log_info "To connect: aws ssm start-session --target ${instance_id} --region ${AWS_REGION}"
    fi
    
    cd ../..
    return 1
  fi
  
  # Launch ResNet50 benchmark job
  log_info "Launching ResNet50 benchmark..."
  local job_resp
  job_resp=$("$RADIX_API" launch-benchmark "${cluster_id}" "${EPOCHS}" "${BATCH_SIZE}")
  
  # Save launch response
  echo "${job_resp}" > "${OBS_DIR}/${trial_id}_launch.json"
  
  log_info "Raw launch response:"
  echo "${job_resp}" | jq '.' || echo "${job_resp}"
  
  # Extract job_id
  local job_id
  job_id=$(echo "${job_resp}" | jq -r '.job.job_id // .job_id // .id // .jobId // empty')
  
  if [ -z "${job_id}" ] || [ "${job_id}" = "null" ]; then
    log_error "Failed to extract job_id from response"
    log_error "Response was: ${job_resp}"
    cleanup_trial "${VPC_ID}" "${SUBNET_ID}" "${AMI_ID}" "${INSTANCE_TYPE}" "${AWS_REGION}" "${run_id}" "${tenant_id}" "${cluster_id}" "${ssm_param_name}"
    cd ../..
    return 1
  fi
  
  log_success "Benchmark job launched: ${job_id}"
  
  # Poll job completion
  log_info "Polling for job completion (max 20 minutes)..."
  local max_poll_seconds=1200
  local start_time
  start_time=$(date +%s)
  local unknown_count=0
  local max_unknown=5
  local job_completed=false
  local throughput=0
  local duration=0
  
  # Build candidate status URLs
  local radix_api_base="${RADIX_API_BASE}"
  local candidates=(
    "${radix_api_base}/v1/jobs/${job_id}"
    "${radix_api_base}/v1/clusters/${cluster_id}/jobs/${job_id}"
    "${radix_api_base}/v1/jobs/${job_id}/status"
  )
  
  # Find working endpoint
  local working_url=""
  log_info "Testing candidate endpoints..."
  for url in "${candidates[@]}"; do
    local safe_url
    safe_url=$(echo "${url}" | sed 's/\?.*$//')
    log_info "Testing: ${safe_url}"
    
    local test_code
    test_code=$(curl -s -o /tmp/endpoint-test.json -w "%{http_code}" \
      -H "Authorization: Bearer ${RADIX_ID_TOKEN}" \
      "${url}" 2>/dev/null || echo "000")
    
    local test_body
    test_body=$(cat /tmp/endpoint-test.json 2>/dev/null | head -c 300)
    log_info "  HTTP ${test_code}: ${test_body:0:100}..."
    
    if [ "${test_code}" != "404" ] && [ "${test_code}" != "000" ]; then
      working_url="${url}"
      log_success "Found working endpoint: ${safe_url}"
      break
    fi
  done
  
  if [ -z "${working_url}" ]; then
    log_error "All candidate endpoints returned 404"
    cleanup_trial "${VPC_ID}" "${SUBNET_ID}" "${AMI_ID}" "${INSTANCE_TYPE}" "${AWS_REGION}" "${run_id}" "${tenant_id}" "${cluster_id}" "${ssm_param_name}"
    cd ../..
    return 1
  fi
  
  # Poll the working endpoint
  while true; do
    local elapsed
    elapsed=$(( $(date +%s) - start_time ))
    
    if [ ${elapsed} -gt ${max_poll_seconds} ]; then
      log_error "Job polling exceeded 20 minute hard limit"
      break
    fi
    
    # Call API and capture HTTP status + body
    local http_code
    http_code=$(curl -s -o "${OBS_DIR}/${trial_id}_status_${elapsed}.json" -w "%{http_code}" \
      -H "Authorization: Bearer ${RADIX_ID_TOKEN}" \
      "${working_url}" || echo "000")
    
    local job_status_json
    job_status_json=$(cat "${OBS_DIR}/${trial_id}_status_${elapsed}.json" 2>/dev/null || echo "{}")
    
    echo "--- Poll iteration (elapsed: ${elapsed}s) ---"
    log_info "HTTP Status: ${http_code}"
    
    # Handle HTTP errors
    if [ "${http_code}" = "404" ]; then
      log_error "Job not found (HTTP 404)"
      break
    elif [ "${http_code}" = "000" ] || [ -z "${job_status_json}" ]; then
      log_error "Failed to reach API or empty response"
      break
    fi
    
    # Extract status (try multiple keys, normalize to lowercase)
    local status
    status=$(echo "${job_status_json}" | jq -r '
      (.status // .state // .job_status // .job.status // .job.state // .job.job_status // empty)
      | if type == "string" then ascii_downcase else empty end
    ')
    
    log_info "Extracted status: '${status}'"
    
    # Check for terminal success states
    if [ "${status}" = "completed" ] || [ "${status}" = "succeeded" ] || [ "${status}" = "success" ] || [ "${status}" = "complete" ]; then
      log_success "Job completed successfully"
      
      # Extract metrics (try both top-level and nested)
      throughput=$(echo "${job_status_json}" | jq -r '.metrics.throughput_images_per_sec // .throughput // .job.metrics.throughput_images_per_sec // .job.throughput // 0')
      duration=$(echo "${job_status_json}" | jq -r '.metrics.duration_seconds // .duration // .job.metrics.duration_seconds // .job.duration // 0')
      
      log_info "Throughput: ${throughput} images/sec"
      log_info "Duration: ${duration} seconds"
      
      job_completed=true
      break
    fi
    
    # Check for terminal failure states
    if [ "${status}" = "failed" ] || [ "${status}" = "error" ] || [ "${status}" = "cancelled" ] || [ "${status}" = "canceled" ]; then
      log_error "Job failed with status: ${status}"
      echo "${job_status_json}" | jq '.' 2>/dev/null || echo "${job_status_json}"
      break
    fi
    
    # Check for running states
    if [ "${status}" = "queued" ] || [ "${status}" = "pending" ] || [ "${status}" = "running" ]; then
      log_info "Job is ${status}, continuing to poll..."
      unknown_count=0
      sleep 15
      continue
    fi
    
    # Unknown status
    if [ -z "${status}" ]; then
      unknown_count=$((unknown_count + 1))
      log_warn "Empty status (count: ${unknown_count}/${max_unknown})"
    else
      unknown_count=$((unknown_count + 1))
      log_warn "Unknown status '${status}' (count: ${unknown_count}/${max_unknown})"
    fi
    
    if [ ${unknown_count} -ge ${max_unknown} ]; then
      log_error "Received unknown/empty status ${max_unknown} times, failing"
      log_error "Last response:"
      echo "${job_status_json}" | jq '.' 2>/dev/null || echo "${job_status_json}"
      break
    fi
    
    sleep 15
  done
  
  # Generate observation file if job completed
  if [ "${job_completed}" = "true" ]; then
    log_info "Generating observation record..."
    
    # Define observation file path
    local obs_file="${OBS_DIR}/${run_id}_${trial_id}_observation.json"
    
    # Generate observation JSON using jq
    jq -n \
      --arg ts "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
      --arg run_id "${run_id}" \
      --arg trial_id "${trial_id}" \
      --arg tenant_id "${tenant_id}" \
      --arg cluster_id "${cluster_id}" \
      --arg job_id "${job_id}" \
      --arg aws_region "${AWS_REGION}" \
      --arg instance_id "${instance_id}" \
      --arg instance_type "${INSTANCE_TYPE}" \
      --argjson throughput "${throughput}" \
      --argjson duration "${duration}" \
      --argjson trial_num "${trial_num}" \
      '{
        ts: $ts,
        run_id: $run_id,
        trial_id: $trial_id,
        tenant_id: $tenant_id,
        cluster_id: $cluster_id,
        job_id: $job_id,
        workload: "resnet50",
        radix_api_base: "${RADIX_API_BASE}",
        instance_type: $instance_type,
        infra: {
          aws_region: $aws_region,
          instance_id: $instance_id,
          instance_type: $instance_type,
          run_id: $run_id
        },
        hardware: {
          gpu_count: 1,
          gpu_model: "unknown"
        },
        outcomes: {
          job_status: "completed",
          throughput_images_per_sec: $throughput,
          duration_seconds: $duration
        },
        trial_number: $trial_num
      }' > "${obs_file}"
    
    log_success "Trial ${trial_num} completed successfully"
    log_info "  Throughput: ${throughput} images/sec"
    log_info "  Duration: ${duration} seconds"
    log_info "WROTE_OBSERVATION=${obs_file}"
    
    SUCCESSFUL_TRIALS=$((SUCCESSFUL_TRIALS + 1))
  else
    log_error "Trial ${trial_num} failed"
    FAILED_TRIALS=$((FAILED_TRIALS + 1))
  fi
  
  # Cleanup infrastructure (unless retain_on_failure is true and trial failed)
  if [ "${job_completed}" = "true" ] || [ "${RETAIN_ON_FAILURE}" != "true" ]; then
    log_info "Destroying EC2 instance..."
    
    terraform destroy -input=false -auto-approve \
      -var="vpc_id=${VPC_ID}" \
      -var="subnet_id=${SUBNET_ID}" \
      -var="ami_id=${AMI_ID}" \
      -var="instance_type=${INSTANCE_TYPE}" \
      -var="region=${AWS_REGION}" \
      -var="run_id=${run_id}" \
      -var="radix_api_base=${RADIX_API_BASE}" \
      -var="tenant_id=${tenant_id}" \
      -var="cluster_id=${cluster_id}" \
      -var="ssm_token_param=${ssm_param_name}"
    log_success "Instance destroyed"
    
    # Cleanup SSM parameter
    aws ssm delete-parameter --name "${ssm_param_name}" --region "${AWS_REGION}" 2>/dev/null || true
  else
    log_warn "Instance retained for debugging (retain_on_failure=true)"
  fi
  
  cd ../..
  echo ""
  
  return 0
}

# Main execution
main() {
  log_info "Starting ${TRIALS} sequential trials"
  log_info "Instance: ${INSTANCE_TYPE}, Batch: ${BATCH_SIZE}, Epochs: ${EPOCHS}"
  echo ""
  
  # Run trials sequentially
  for i in $(seq 1 "${TRIALS}"); do
    if ! run_trial "${i}"; then
      FAILED_TRIALS=$((FAILED_TRIALS + 1))
    fi
  done
  
  # Summary
  echo "=========================================="
  echo "Evidence Pack Summary"
  echo "=========================================="
  echo "Total trials: ${TRIALS}"
  echo "Successful: ${SUCCESSFUL_TRIALS}"
  echo "Failed: ${FAILED_TRIALS}"
  echo ""
  
  # List observations found
  log_info "Listing observations directory:"
  ls -la "$OBS_DIR" || true
  echo ""
  
  # Count observation files
  OBS_COUNT=$(ls -1 "$OBS_DIR"/*_observation.json 2>/dev/null | wc -l | tr -d ' ')
  log_info "Observation files found: ${OBS_COUNT}"
  
  # Save trial counts and observation count for summary generation
  mkdir -p "${ROOT_DIR}/artifacts"
  echo "${OBS_COUNT}" > "${ROOT_DIR}/artifacts/observations_count.txt"
  echo "${SUCCESSFUL_TRIALS}" > "${ROOT_DIR}/artifacts/successful_trials.txt"
  echo "${FAILED_TRIALS}" > "${ROOT_DIR}/artifacts/failed_trials.txt"
  
  # Exit with error if no observations were produced
  if [ "${OBS_COUNT}" -eq 0 ]; then
    log_error "ERROR: 0 observations produced; see logs above."
    exit 1
  fi
  
  log_success "Evidence pack trials completed with ${OBS_COUNT} observations"
}

# Run main function
main
