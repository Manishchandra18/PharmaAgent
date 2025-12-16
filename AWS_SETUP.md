# AWS Textract Setup Instructions

## Prerequisites
1. AWS Account with Textract access
2. AWS IAM user with Textract permissions

## Required Environment Variables

Add these to your `.env` file in the backend directory:

```bash
# AWS Credentials for Textract
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1

# Optional: AWS Session Token (if using temporary credentials)
# AWS_SESSION_TOKEN=your_session_token_here
```

## AWS IAM Permissions

Your AWS user/role needs the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "textract:DetectDocumentText",
                "textract:AnalyzeDocument"
            ],
            "Resource": "*"
        }
    ]
}
```

## AWS Textract Benefits

- **Higher Accuracy**: Better than Tesseract for complex layouts
- **Form Recognition**: Automatically detects forms and tables
- **Handwriting Support**: Excellent for handwritten prescriptions
- **Confidence Scores**: Built-in confidence scoring
- **Medical Documents**: Optimized for medical forms and prescriptions

## Fallback Mechanism

If AWS Textract is not configured or fails, the system automatically falls back to Tesseract OCR, ensuring the application continues to work.

## Cost Considerations

AWS Textract pricing (as of 2024):
- First 1,000 pages per month: Free
- Additional pages: $1.50 per 1,000 pages

For development and testing, the free tier should be sufficient.
