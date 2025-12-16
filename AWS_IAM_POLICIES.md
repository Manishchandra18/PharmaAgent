# AWS IAM Policies for Medical AI Agent

## Option 1: Minimal Textract-Only Policy (Recommended for Production)

Create a custom policy with only the necessary Textract permissions:

### Policy Name: `MedicalAI-Textract-Policy`

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "TextractDocumentAnalysis",
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

## Option 2: Comprehensive Policy (For Development/Testing)

If you want broader permissions for testing and development:

### Policy Name: `MedicalAI-Development-Policy`

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "TextractFullAccess",
            "Effect": "Allow",
            "Action": [
                "textract:*"
            ],
            "Resource": "*"
        },
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "*"
        }
    ]
}
```

## Option 3: AWS Managed Policy (Simplest)

Use the AWS managed policy:

### Policy: `AmazonTextractFullAccess`

This is an AWS-managed policy that provides full access to Textract services.

## Step-by-Step IAM Setup

### 1. Create IAM User

1. Go to AWS Console → IAM → Users
2. Click "Create user"
3. Username: `medical-ai-textract-user`
4. Select "Programmatic access" (for API keys)
5. Click "Next"

### 2. Attach Policies

Choose one of the following approaches:

#### Approach A: Custom Policy (Recommended)
1. Click "Attach policies directly"
2. Click "Create policy"
3. Use the JSON from Option 1 above
4. Name: `MedicalAI-Textract-Policy`
5. Create and attach to user

#### Approach B: AWS Managed Policy (Easier)
1. Click "Attach policies directly"
2. Search for "AmazonTextractFullAccess"
3. Select and attach

### 3. Create Access Keys

1. Go to the user you created
2. Click "Security credentials" tab
3. Click "Create access key"
4. Choose "Application running outside AWS"
5. Copy the Access Key ID and Secret Access Key
6. **Important:** Save these securely - you won't be able to see the secret key again

### 4. Add to Environment Variables

Add these to your `.env` file:

```bash
AWS_ACCESS_KEY_ID=AKIA... (your access key)
AWS_SECRET_ACCESS_KEY=... (your secret key)
AWS_REGION=us-east-1
```

## Security Best Practices

### 1. Principle of Least Privilege
- Use Option 1 (minimal policy) for production
- Only grant permissions that are actually needed

### 2. Environment-Specific Policies
- Development: Use broader permissions for testing
- Production: Use minimal permissions for security

### 3. Regular Key Rotation
- Rotate access keys every 90 days
- Use AWS IAM Access Key Last Used to monitor usage

### 4. Monitor Usage
- Enable CloudTrail to log API calls
- Set up billing alerts for Textract usage

## Cost Monitoring

### Set up Billing Alerts:
1. Go to AWS Billing → Budgets
2. Create budget for Textract costs
3. Set alert threshold (e.g., $10/month)

### Textract Pricing (as of 2024):
- **Free Tier:** 1,000 pages/month
- **Additional:** $1.50 per 1,000 pages
- **Forms/Tables:** $0.50 per 1,000 pages (analyze_document)

## Troubleshooting

### Common Issues:

1. **Access Denied Error:**
   - Check if IAM policy is attached
   - Verify access keys are correct
   - Ensure region is correct

2. **Invalid Credentials:**
   - Regenerate access keys
   - Check .env file format
   - Restart the application

3. **Region Mismatch:**
   - Ensure AWS_REGION matches your Textract service region
   - Default regions: us-east-1, us-west-2, eu-west-1

### Test Your Setup:

Run this command to test your AWS credentials:

```bash
aws textract detect-document-text --document '{"Bytes":"base64-encoded-image"}' --region us-east-1
```

## Production Recommendations

1. **Use IAM Roles** instead of access keys when possible
2. **Enable MFA** on the IAM user
3. **Set up CloudWatch** monitoring
4. **Use AWS Secrets Manager** for credential storage
5. **Implement proper error handling** for API failures
