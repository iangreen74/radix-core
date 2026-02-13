package main

# Default: deny all K8s objects unless explicitly enabled via annotation.
deny[msg] {
  input.kind != ""
  not input.metadata.annotations["radix/deploy-enabled"]
  msg := sprintf("deploy disabled by policy for %s/%s", [input.kind, input.metadata.name])
}
