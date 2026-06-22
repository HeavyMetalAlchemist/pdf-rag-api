variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "s3_bucket_name" {
  description = "S3 bucket for document storage"
  type        = string
}

variable "s3_vector_bucket_name" {
  description = "S3 Vectors bucket for embeddings"
  type        = string
}

variable "s3_vector_index_name" {
  description = "S3 Vectors index name"
  type        = string
}

