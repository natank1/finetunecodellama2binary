from web3 import Web3
import pickle
import time
def get_logs_of_tx(transaction_receipt):
    function_names = []
    for log in transaction_receipt.logs:
        if log["topics"][0] == "0x...":  # This is the topic that indicates a function call
            function_name = log["topics"][1].hex()[2:]  # This is the topic that contains the function name
            function_names.append(function_name)
    return function_names

def wallet_or_contract(tx_hash):
    is_from_contract = web3.eth.get_code(tx_hash) != b''
    return is_from_contract
def check_validiy(tx,black_list):
    if tx['to'] is None :
        return False
    if tx['from'] is None :
        return False
    if tx['to']   in black_list:
        return False


    if tx['from'] in black_list:
        return False
    return True
infurakey='hhh'
infura_url = 'https://mainnet.infura.io/v3/'+infurakey

# Initialize a connection to an Ethereum node
web3 = Web3(Web3.HTTPProvider(infura_url))

# Replace 'YOUR_CONTRACT_ADDRESS' with the actual contract address you're interested in
contract_address = 'YOUR_CONTRACT_ADDRESS'

from web3 import Web3
def wallet_or_contract(tx_hash,web3):
    is_from_contract = web3.eth.get_code(tx_hash) != b''
    return is_from_contract
if __name__ == '__main__':
    # Create an instance of the contract
    # contract = web3.eth.contract(address=web3.to_checksum_address(contract_address))

    # Get the total number of transactions related to the contract
    # block_number = web3.eth.block_number
    # print (block_number)
    contract_transactions = []
    # with open("/home/ubuntu/data/all_black.pkl", "rb") as file:
    #     xx = pickle.load(file)
    # black_list =set(xx)
    # print (len(black_list))
    print ('yyyyy')
    with open("good_trans_all.pkl", "rb") as file:
        all_t0 = pickle.load(file)
    allt00 ={i[0]:(i[1]['from'],i[1]['to']) for i in all_t0.items()}
    all_t0={}
    with open("new_good_trans0.pkl", "rb") as file:
        all_t1 = pickle.load(file)

    print (len(all_t1))
    # Iterate through all blocks and transactions to find relevant ones
    print (len(all_t0))
    # all_t1 ={}
    for cnt,block_t in enumerate(allt00):

        trans_name =   block_t
        if trans_name in all_t1:
            if cnt%500==399:
                print (cnt)
            continue

        transaction_receipt = web3.eth.get_transaction_receipt(trans_name)

        is_from_contr =wallet_or_contract(allt00[trans_name][0])
        is_to_contr = wallet_or_contract(allt00[trans_name][1])

        # funcion_names =get_logs_of_tx(transaction_receipt)
        xx= (is_from_contr,is_to_contr)
        all_t1.setdefault(trans_name,xx)


        time.sleep(2)



    #         # if tx['to'] is not None and tx['to'].lower() == contract_address.lower():
    #             contract_transactions.append(tx)
    #
    # # Now, contract_transactions contains all transactions related to the contract
    # print(contract_transactions)
    # # Make sure to replace 'YOUR_INFURA_PROJECT_ID' with your Infura project ID and 'YOUR_CONTRACT_ADDRESS' with the address of the smart contract you want to retrieve transactions for.
    # #
    # # This code connects to an Ethereum node using Infura and iterates through all blocks on the Ethereum blockchain, filtering transactions that have the specified contract address in the to field. The relevant transactions are stored in the contract_transactions list.
    # #
    # # Please note that this code may take some time to run, especially if you are analyzing a large number of blocks. Additionally, you should handle rate limiting and pagination when working with large datasets.
    # #
    # #
    # #
    # #
    # # Is this conversation helpful so far?
    #


