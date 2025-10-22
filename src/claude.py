import boto3
import json



print("Available Anthropic models in your AWS account:")
print("=" * 60)

bedrock = boto3.client(service_name="bedrock", region_name="us-east-1")
response = bedrock.list_foundation_models(byProvider="anthropic")

print("Available Anthropic models in your AWS account:")
print("=" * 60)

for summary in response["modelSummaries"]:
    print(f"Model ID: {summary['modelId']}")
    print(f"Model Name: {summary.get('modelName', 'N/A')}")
    print(f"Input Modalities: {', '.join(summary.get('inputModalities', []))}")
    print(f"Output Modalities: {', '.join(summary.get('outputModalities', []))}")
    print(f"Model Lifecycle Status: {summary.get('modelLifecycle', {}).get('status', 'N/A')}")
    print("-" * 60)
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
body = json.dumps({
  "max_tokens": 256,
  "messages": [{"role": "user", "content": "Hello, world"}],
  "anthropic_version": "bedrock-2023-05-31"
})

response = bedrock.invoke_model(body=body, modelId="us.anthropic.claude-sonnet-4-20250514-v1:0")

response_body = json.loads(response.get("body").read())
print(response_body.get("content"))


'''
Available Anthropic models in your AWS account:
============================================================
Model ID: anthropic.claude-sonnet-4-20250514-v1:0
Model Name: Claude Sonnet 4
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-opus-4-1-20250805-v1:0
Model Name: Claude Opus 4.1
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-instant-v1:2:100k
Model Name: Claude Instant
Input Modalities: TEXT
Output Modalities: TEXT
Model Lifecycle Status: LEGACY
------------------------------------------------------------
Model ID: anthropic.claude-v2:0:18k
Model Name: Claude
Input Modalities: TEXT
Output Modalities: TEXT
Model Lifecycle Status: LEGACY
------------------------------------------------------------
Model ID: anthropic.claude-v2:0:100k
Model Name: Claude
Input Modalities: TEXT
Output Modalities: TEXT
Model Lifecycle Status: LEGACY
------------------------------------------------------------
Model ID: anthropic.claude-v2:1:18k
Model Name: Claude
Input Modalities: TEXT
Output Modalities: TEXT
Model Lifecycle Status: LEGACY
------------------------------------------------------------
Model ID: anthropic.claude-v2:1:200k
Model Name: Claude
Input Modalities: TEXT
Output Modalities: TEXT
Model Lifecycle Status: LEGACY
------------------------------------------------------------
Model ID: anthropic.claude-3-sonnet-20240229-v1:0:28k
Model Name: Claude 3 Sonnet
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: LEGACY
------------------------------------------------------------
Model ID: anthropic.claude-3-sonnet-20240229-v1:0:200k
Model Name: Claude 3 Sonnet
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: LEGACY
------------------------------------------------------------
Model ID: anthropic.claude-3-sonnet-20240229-v1:0
Model Name: Claude 3 Sonnet
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: LEGACY
------------------------------------------------------------
Model ID: anthropic.claude-3-haiku-20240307-v1:0:48k
Model Name: Claude 3 Haiku
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-3-haiku-20240307-v1:0:200k
Model Name: Claude 3 Haiku
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-3-haiku-20240307-v1:0
Model Name: Claude 3 Haiku
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-3-opus-20240229-v1:0:12k
Model Name: Claude 3 Opus
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-3-opus-20240229-v1:0:28k
Model Name: Claude 3 Opus
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-3-opus-20240229-v1:0:200k
Model Name: Claude 3 Opus
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-3-opus-20240229-v1:0
Model Name: Claude 3 Opus
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-3-5-sonnet-20240620-v1:0
Model Name: Claude 3.5 Sonnet
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-3-5-sonnet-20241022-v2:0
Model Name: Claude 3.5 Sonnet v2
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-3-7-sonnet-20250219-v1:0
Model Name: Claude 3.7 Sonnet
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-3-5-haiku-20241022-v1:0
Model Name: Claude 3.5 Haiku
Input Modalities: TEXT
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
------------------------------------------------------------
Model ID: anthropic.claude-opus-4-20250514-v1:0
Model Name: Claude Opus 4
Input Modalities: TEXT, IMAGE
Output Modalities: TEXT
Model Lifecycle Status: ACTIVE
'''


