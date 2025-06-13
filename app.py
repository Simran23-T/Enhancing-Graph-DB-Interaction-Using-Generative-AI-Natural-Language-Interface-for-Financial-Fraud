import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from neo4j import GraphDatabase
import re
from pyvis.network import Network
import os

# Corrected paths for model and tokenizer
saved_model_path = "/home/ritcse/Drive/BBB/simran-20240524T044553Z-001/simran/Graph"

# Load the saved T5 model and tokenizer
try:
    model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
    tokenizer = T5Tokenizer.from_pretrained(saved_model_path)
except Exception as e:
    st.error(f"Error loading model/tokenizer: {e}")
    print(f"Error loading model/tokenizer: {e}")

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def execute_query(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return result.data()

# Initialize Neo4j connection
neo_uri = "bolt://localhost:7687"
neo_user = "neo4j"
neo_password = "12345678"
neo_connector = Neo4jConnector(neo_uri, neo_user, neo_password)

def generate_predictions(model, tokenizer, input_texts, max_length=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    inputs = tokenizer.batch_encode_plus(input_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def generate_cypher_query(input_text):
    transfer_match = re.search(r"transfer of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.", input_text, re.IGNORECASE)
    cash_out_match = re.search(r"cash_out of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.", input_text, re.IGNORECASE)
    cash_in_match = re.search(r"cash_in of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.", input_text, re.IGNORECASE)
    payment_match = re.search(r"payment of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.", input_text, re.IGNORECASE)

    if transfer_match:
        amount = transfer_match.group(1)
        account_number = transfer_match.group(2)
        old_balance = transfer_match.group(3)
        new_balance = transfer_match.group(4)

        cypher_query = f"""
        MATCH (t:Transaction)
        WHERE t.type = 'TRANSFER' AND toFloat(t.amount) = {amount} AND t.nameOrig = '{account_number}' 
        AND toFloat(t.oldbalanceOrg) = {old_balance} AND toFloat(t.newbalanceOrig) = {new_balance} 
        AND toFloat(t.oldbalanceDest) = 0.0 AND toFloat(t.newbalanceDest) = 0.0
        RETURN t
        """
    elif cash_out_match:
        amount = cash_out_match.group(1)
        account_number = cash_out_match.group(2)
        old_balance = cash_out_match.group(3)
        new_balance = cash_out_match.group(4)

        cypher_query = f"""
        MATCH (t:Transaction)
        WHERE t.type = 'CASH_OUT' AND toFloat(t.amount) = {amount} AND t.nameOrig = '{account_number}' 
        AND toFloat(t.oldbalanceOrg) = {old_balance} AND toFloat(t.newbalanceOrig) = {new_balance}
        RETURN t
        """
    elif cash_in_match:
        amount = cash_in_match.group(1)
        account_number = cash_in_match.group(2)
        old_balance = cash_in_match.group(3)
        new_balance = cash_in_match.group(4)

        cypher_query = f"""
        MATCH (t:Transaction)
        WHERE t.type = 'CASH_IN' AND toFloat(t.amount) = {amount} AND t.nameOrig = '{account_number}' 
        AND toFloat(t.oldbalanceOrg) = {old_balance} AND toFloat(t.newbalanceOrig) = {new_balance}
        RETURN t
        """
    elif payment_match:
        amount = payment_match.group(1)
        account_number = payment_match.group(2)
        old_balance = payment_match.group(3)
        new_balance = payment_match.group(4)

        cypher_query = f"""
        MATCH (t:Transaction)
        WHERE t.type = 'PAYMENT' AND toFloat(t.amount) = {amount} AND t.nameOrig = '{account_number}' 
        AND toFloat(t.oldbalanceOrg) = {old_balance} AND toFloat(t.newbalanceOrig) = {new_balance}
        RETURN t
        """
    else:
        cypher_query = None

    return cypher_query

def visualize_network(neo_result):
    net = Network(height="500px", width="100%", notebook=False)
    net.show_buttons(filter_=['physics'])

    for record in neo_result:
        t = record['t']
        
        # Add nodes with labels indicating the sender and receiver
        sender_label = f"{t['nameOrig']} (Sent: ${t['amount']}, Type: {t['type']}, Old Balance: ${t['oldbalanceOrg']}, New Balance: ${t['newbalanceOrig']})"
        receiver_label = f"{t['nameDest']} (Received: ${t['amount']}, Type: {t['type']}, Old Balance: ${t['oldbalanceDest']}, New Balance: ${t['newbalanceDest']})"

        if t['isFraud'] == 1:
            # Highlight fraudulent transactions in red
            net.add_node(t['nameOrig'], label=sender_label, color='red')
            net.add_node(t['nameDest'], label=receiver_label, color='red')
        else:
            net.add_node(t['nameOrig'], label=sender_label)
            net.add_node(t['nameDest'], label=receiver_label)
        
        # Add edge with details like transaction type and amount
        edge_label = f"{t['type']} (${t['amount']})"
        net.add_edge(t['nameOrig'], t['nameDest'], label=edge_label)

    return net

# Streamlit UI
st.title('Financial Fraud Detection System')

# Sidebar for user input
st.sidebar.title('User Input')
user_query = st.sidebar.text_area('Enter your query in natural language', placeholder='Example: "Please check whether the transfer of $420330.71 from account C1868228472 might be fraudulent. It started with a balance of $420330.71 and ended with $0.0."')

if st.sidebar.button('Execute Query'):
    if user_query:
        try:
           

            cypher_query = generate_cypher_query(user_query)
            st.write("Generated Cypher Query:")
            st.write(cypher_query)  # Display the generated Cypher query for debugging

            # Check if Cypher query is generated correctly
            if cypher_query:
                neo_result = neo_connector.execute_query(cypher_query)
                st.write("Neo4j Result:")
                st.write(neo_result)  # Display the raw result for debugging

                if neo_result:
                    # Display transaction records with expandable sections
                    for index, record in enumerate(neo_result):
                        st.write(f"Record {index + 1}:")
                        with st.expander(f"Transaction Details {index + 1}"):
                            t = record['t']
                            st.write(f"Transaction Type: {t['type']}")
                            st.write(f"Amount: ${t['amount']}")
                            st.write(f"Sender Account: {t['nameOrig']}")
                            st.write(f"Recipient Account: {t['nameDest']}")
                            st.write(f"Fraud Status: {'Fraudulent' if t['isFraud'] == 1 else 'Not Fraudulent'}")  # Correctly check for integer value
                            st.write(f"Raw Record: {t}")  # Display raw record for debugging

                    net = visualize_network(neo_result)
                    html_file_path = "neo4j_graph.html"
                    net.save_graph(html_file_path)

                    if os.path.exists(html_file_path):
                        # Display the HTML file within Streamlit
                        with open(html_file_path, "r") as f:
                            html_content = f.read()
                        components.html(html_content, height=600)
                    else:
                        st.write("Error: The HTML file was not created successfully.")
                else:
                    st.write("No result found in Neo4j.")
            else:
                st.write("Cypher query could not be generated. Please check the input format.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


