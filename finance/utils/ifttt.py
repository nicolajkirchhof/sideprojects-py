import requests
import json

def send_ifttt_webhook(event_name, api_key, values=None):
  """
  Send a webhook request to IFTTT Maker Webhooks

  Args:
      event_name (str): The name of your IFTTT webhook event
      api_key (str): Your IFTTT Maker Webhooks key
      values (list): Optional dictionary containing up to 3 values [value1, value2, value3]

  Returns:
      requests.Response: Response from the IFTTT server
  """
  # Base URL for IFTTT webhooks
  url = f"https://maker.ifttt.com/trigger/{event_name}/with/key/{api_key}"

  # Default headers
  headers = {
    'Content-Type': 'application/json',
  }

  # Prepare the payload
  payload = {}
  if values:
    # IFTTT accepts up to 3 values named value1, value2, value3
    for i, value in enumerate(values, 1):
      if i <= 3:  # Only use first 3 values
        payload[f'value{i}'] = str(value)

  try:
    # Send POST request to IFTTT
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise exception for bad status codes

    print(f"Webhook sent successfully! Status code: {response.status_code}")
    return response

  except requests.exceptions.RequestException as e:
    print(f"Error sending webhook: {e}")
    return None
