"""
Graph schema definition for IEEE-CIS Fraud Detection.

Node types:
- Transaction
- Card
- Email
- Device
- Address

Edge relationships:
- Card -> Transaction
- Email -> Transaction
- Device -> Transaction
- Address -> Transaction
"""

NODE_TYPES = {
    "transaction": "Transaction",
    "card": "Card",
    "email": "Email",
    "device": "Device",
    "address": "Address",
}


EDGE_TYPES = {
    "card_transaction": ("Card", "Transaction"),
    "email_transaction": ("Email", "Transaction"),
    "device_transaction": ("Device", "Transaction"),
    "address_transaction": ("Address", "Transaction"),
}


def describe_schema():

    print("\n========== GRAPH SCHEMA ==========\n")

    print("Node Types:")

    for node in NODE_TYPES.values():
        print("-", node)

    print("\nRelationships:")

    for edge, relation in EDGE_TYPES.items():
        print(f"- {relation[0]} --> {relation[1]}")

    print("\n==================================")


if __name__ == "__main__":
    describe_schema()
