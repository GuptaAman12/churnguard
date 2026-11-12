output "aws_account_id" {
  description = "AWS account hosting the stack."
  value       = data.aws_caller_identity.current.account_id
}

output "alb_dns_name" {
  description = "Public DNS name of the prediction API load balancer."
  value       = aws_lb.api.dns_name
}

output "ecr_repository_urls" {
  description = "ECR repositories to push the API and MLflow images to."
  value       = { for key, repo in aws_ecr_repository.app : key => repo.repository_url }
}

output "mlflow_artifacts_bucket" {
  description = "S3 bucket backing the MLflow artifact store."
  value       = aws_s3_bucket.mlflow.id
}

output "mlflow_internal_dns" {
  description = "Cloud Map DNS name used by the API to reach MLflow."
  value       = local.mlflow_dns
}

output "mlflow_rds_endpoint" {
  description = "Private RDS endpoint for the MLflow backend store."
  value       = aws_db_instance.mlflow.address
}

output "api_secret_arn" {
  description = "Secrets Manager ARN containing CHURNGUARD_API_KEY."
  value       = aws_secretsmanager_secret.app.arn
}

output "cloudwatch_log_groups" {
  description = "Log groups for the API and MLflow services."
  value = {
    api    = aws_cloudwatch_log_group.api.name
    mlflow = aws_cloudwatch_log_group.mlflow.name
  }
}
