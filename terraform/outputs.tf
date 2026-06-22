output "s3_bucket_name" {
  description = "S3 bucket for PDF storage"
  value       = aws_s3_bucket.documents.bucket
}

output "s3_vector_bucket_name" {
  description = "S3 Vectors bucket name"
  value       = aws_s3vectors_vector_bucket.this.vector_bucket_name
}

output "s3_vector_index_name" {
  description = "S3 Vectors index name"
  value       = aws_s3vectors_index.documents.index_name
}