terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.24"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------
# Regular S3 bucket — stores original uploaded PDFs
# ---------------------------------------------------------------
resource "aws_s3_bucket" "documents" {
  bucket = var.s3_bucket_name

  # Set to true to prevent accidental removal of documents
  lifecycle {
    prevent_destroy = false
  }

  tags = {
    Project   = "document-intelligence-api"
    ManagedBy = "terraform"
  }
}

# Block all public access — documents are private
resource "aws_s3_bucket_public_access_block" "documents" {
  bucket = aws_s3_bucket.documents.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------
# S3 Vectors bucket — stores chunk embeddings
# ---------------------------------------------------------------
resource "aws_s3vectors_vector_bucket" "this" {
  vector_bucket_name = var.s3_vector_bucket_name
}

# ---------------------------------------------------------------
# S3 Vectors index — cosine similarity, 1024 dim (Titan Text v2)
# ---------------------------------------------------------------
resource "aws_s3vectors_index" "documents" {
  vector_bucket_name = aws_s3vectors_vector_bucket.this.vector_bucket_name
  index_name         = var.s3_vector_index_name
  data_type          = "float32"
  dimension          = 1024
  distance_metric    = "cosine"
}