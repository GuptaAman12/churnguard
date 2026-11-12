variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project slug used to prefix resource names."
  type        = string
  default     = "churnguard"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,20}$", var.project_name))
    error_message = "project_name must be 3-21 lowercase alphanumeric characters or hyphens."
  }
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)."
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging or prod."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.40.0.0/16"
}

variable "availability_zone_count" {
  description = "Number of AZs to spread subnets across."
  type        = number
  default     = 2

  validation {
    condition     = var.availability_zone_count >= 2 && var.availability_zone_count <= 4
    error_message = "availability_zone_count must be between 2 and 4."
  }
}

variable "api_image_tag" {
  description = "Container image tag for the prediction API (Git SHA recommended)."
  type        = string
  default     = "latest"
}

variable "mlflow_image_tag" {
  description = "Container image tag for the MLflow server."
  type        = string
  default     = "latest"
}

variable "desired_count" {
  description = "Desired number of API tasks."
  type        = number
  default     = 2
}

variable "task_cpu" {
  description = "CPU units for the API task (1024 = 1 vCPU)."
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Memory (MiB) for the API task."
  type        = number
  default     = 1024
}

variable "mlflow_task_cpu" {
  description = "CPU units for the MLflow task."
  type        = number
  default     = 1024
}

variable "mlflow_task_memory" {
  description = "Memory (MiB) for the MLflow task."
  type        = number
  default     = 2048
}

variable "db_instance_class" {
  description = "RDS instance class for the MLflow backend store."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage" {
  description = "Initial RDS storage in GiB."
  type        = number
  default     = 20
}

variable "certificate_arn" {
  description = "Optional ACM certificate ARN. When set, HTTP is redirected to HTTPS."
  type        = string
  default     = ""
}

variable "skip_final_snapshot" {
  description = "Skip the final RDS snapshot on destroy (dev only)."
  type        = bool
  default     = true
}

variable "force_destroy_bucket" {
  description = "Allow Terraform to delete the MLflow artifact bucket and its objects."
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days."
  type        = number
  default     = 30
}
