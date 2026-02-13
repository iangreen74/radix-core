{{/*
Expand the name of the chart.
*/}}
{{- define "scheduler-webhook.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "scheduler-webhook.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "scheduler-webhook.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "scheduler-webhook.labels" -}}
helm.sh/chart: {{ include "scheduler-webhook.chart" . }}
{{ include "scheduler-webhook.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/component: webhook
app.kubernetes.io/part-of: radix
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "scheduler-webhook.selectorLabels" -}}
app.kubernetes.io/name: {{ include "scheduler-webhook.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "scheduler-webhook.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "scheduler-webhook.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate certificate name
*/}}
{{- define "scheduler-webhook.certificateName" -}}
{{- printf "%s-tls" (include "scheduler-webhook.fullname" .) }}
{{- end }}

{{/*
Generate webhook name
*/}}
{{- define "scheduler-webhook.webhookName" -}}
{{- printf "%s.%s.svc" (include "scheduler-webhook.fullname" .) .Release.Namespace }}
{{- end }}
