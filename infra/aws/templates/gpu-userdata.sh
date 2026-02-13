#!/bin/bash
# GPU Node User Data for Radix EKS Cluster

set -o xtrace

# Bootstrap the node
/etc/eks/bootstrap.sh ${cluster_name} \
  --b64-cluster-ca ${ca_data} \
  --apiserver-endpoint ${endpoint} \
  --container-runtime containerd \
  --kubelet-extra-args '--node-labels=radix.ai/gpu-node=true,radix.ai/instance-type=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)'

# Install NVIDIA drivers and container runtime
yum update -y
yum install -y gcc kernel-devel-$(uname -r)

# Install NVIDIA drivers
aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
chmod +x NVIDIA-Linux-x86_64*.run
./NVIDIA-Linux-x86_64*.run --silent --dkms

# Install nvidia-container-runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | \
  tee /etc/yum.repos.d/libnvidia-container.repo

yum install -y nvidia-container-runtime

# Configure containerd for GPU support
mkdir -p /etc/containerd
cat > /etc/containerd/config.toml << EOF
version = 2
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  privileged_without_host_devices = false
  runtime_engine = ""
  runtime_root = ""
  runtime_type = "io.containerd.runc.v2"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
    BinaryName = "/usr/bin/nvidia-container-runtime"
EOF

systemctl restart containerd

# Verify GPU setup
nvidia-smi

echo "GPU node setup complete"
