package main

# Deny any pod/container requesting GPUs unless explicitly allowed.
deny[msg] {
  some c
  c := input.spec.template.spec.containers[_]
  res := c.resources.limits["nvidia.com/gpu"]
  res != null
  not input.metadata.annotations["radix.allow.gpu"]
  msg := sprintf("GPU request denied (set radix.allow.gpu to allow): %s", [input.metadata.name])
}
