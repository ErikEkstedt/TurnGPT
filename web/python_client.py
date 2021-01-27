import requests
import json


sample_url = "http://localhost:5000/sample"
prediction_url = "http://localhost:5000/prediction"
response_url = "http://localhost:5000/response_ranking"
trp_url = "http://localhost:5000/trp"

##################################################################
# Sample response
template = {
    "text": ["Hello there, how can I help you?", "yes hi i would like to order a pizza"]
}
response = requests.post(sample_url, json=template)
d = json.loads(response.content.decode())
print("Sample: ", d)

##################################################################
# TRP response
template = {
    "text": [
        "Hello there, how can I help you?",
        "yes hi i would like to order a pizza please",
    ]
}
response = requests.post(trp_url, json=template)
d = json.loads(response.content.decode())
print("EOT: ", d)

##################################################################
# Prediction
template = {
    "text": ["Hello there, how can I help you?", "yes hi i would like to order a pizza"]
}
response = requests.post(prediction_url, json=template)
d = json.loads(response.content.decode())
print(d)


##################################################################
# Rank Response
template = {
    "text": [
        "Hello there, how can I help you?",
        "yes hi i would like to order a pizza",
    ],
    "responses": [
        "from where would you like to order?",
        "what toppings would you like?",
        "for how many people?",
        "I can help you with that",
        "Would you like pepperoni or cheese?",
        "There are many places available, from where would you like to order?",
        "okay, hold on",
        "okay, I can help you with that",
    ],
}
response = requests.post(response_url, json=template)
d = json.loads(response.content.decode())
for r in template["responses"]:
    print(r)
print("BEST: ", d)
