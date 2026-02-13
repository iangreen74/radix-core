
{{- define "radix_core.observerImage" -}}
{{- if .Values.images.pinDigests -}}
{{- printf "%s@%s" (default "ghcr.io/iangreen74/radix-observer" .Values.images.observer.repository) (default "sha256:000" .Values.images.observer.digest) -}}
{{- else -}}
{{- printf "%s:%s" (default "ghcr.io/iangreen74/radix-observer" .Values.images.observer.repository) (default "latest" .Values.images.observer.tag) -}}
{{- end -}}
{{- end -}}
