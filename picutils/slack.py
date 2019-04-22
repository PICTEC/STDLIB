import json
import os
import requests

class SlackNotifications:
    def __init__(self, token=None):
        if token is not None:
            self.token = token
        elif os.path.isfile(".slack.token"):
            self.token = open(".slack.token").read().strip()
        elif os.path.isfile(os.expanduser("~/.local/.slack.token")):
            self.token = open(os.expanduser("~/.local/.slack.token")).read()
        else:
            raise ValueError("Token not present and cannot be found in config files")

    def notify(self, message):
        headers = {
            'Content-type': 'application/json',
        }
        data = json.dumps({"text": message})
        return requests.post(
            self.token,
            headers=headers, data=data)
