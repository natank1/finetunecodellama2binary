import pickle
import numpy as np
import torch
convert_to_gwei = 1/ pow(10,9)
#gasprice and effectivegasprice are always equal
from decoding_sign.function_features import octet_process
octet_ob= octet_process()

def extract_functions_from_input_data(web3, contract_address, transaction ):


    # Get the input data from the transaction
    input_data = transaction['input'].hex()[2:]  # Remove '0x' prefix

    # Attempt to find function signatures that match the input data
    #

    for i in range(0, len(input_data), 8):  # Assuming each function selector is 4 bytes (8 characters)
        function_selector=input_data[i:i+8]

        # Try to decode the input data using the function selector
        try:

            zz =web3.eth.contract(contract_address, abi=[])
            decoded_data = web3.eth.contract(contract_address, abi=[]).functions.decode_input_data(function_selector, input_data)
            print(f"Function call detected: {decoded_data}")
            print ("yes")
        except:
            print(f"Unable to decode data for function selector: {function_selector}")
    return
def check_address_is_contract(address,w3):
    # Connect to an Ethereum node

    # checksum_address = Web3.to_checksum_address(address)
    # Get the code of the address
    code = w3.eth.get_code(address)

    # Check if the code is empty
    if len(code) == 0:
        # If the code is empty, the address is not a contract
        return False,'',''
    else:
        abi1 = code.hex()[8:]
        abi=w3.eth.contract(bytecode=abi1).abi
        # If the code is not empty, the address is a contract
        return True,code,abi



def collect_topics(logs):

    mega_set= set()
    for log in logs:
        mm =log['topics']
        for m in mm:
            mega_set.add(m)
    return mega_set
def geneate_feaure_for_single_tx(w3,dic_val,rec_ind,tx_ind,label,list_of_topics,outpfile,save_names=False ):
    names_list =[]
    list_of_feautes= []
    receipt_val =dic_val[rec_ind]
    tr_val = dic_val[tx_ind]

    # extract_functions_from_input_data(w3, tr_val['to'],tr_val)
    gw_effective =convert_to_gwei*receipt_val['effectiveGasPrice']
    list_of_feautes.append(gw_effective)
    names_list.append('effectiveGasPrice_gwei')
    # isaddress = int(receipt_val['contractAddress'] is None)
    # names_list.append('isaddress')
    # list_of_feautes.append(isaddress)
    list_of_feautes.append(receipt_val['cumulativeGasUsed'])
    names_list.append('cumulGused')
    # if 'status' in receipt_val:
    #         x= receipt_val['status']
    # else:
    #         x=-1
    # list_of_feautes.append(x)
    # names_list.append('status')
    if tr_val['gas']>0 and receipt_val['gasUsed']>0:
        list_of_feautes.append(receipt_val['gasUsed']/tr_val['gas'])
    else:
        list_of_feautes.append(-1)
    names_list.append('gas_ratio')

    # if receipt_val['gasUsed'] > 0 and receipt_val['cumulativeGasUsed'] > 0:
    #     list_of_feautes.append(receipt_val['gasUsed'] / (receipt_val['cumulativeGasUsed'] +receipt_val['gasUsed']))
    # else:
    #     list_of_feautes.append(-1)
    # names_list.append('gcratious')

    # if tr_val['gas'] > 0 and  receipt_val['cumulativeGasUsed'] > 0:
    #     list_of_feautes.append( tr_val['gas']/(tr_val['gas']+receipt_val['cumulativeGasUsed']))
    # else:
    #     list_of_feautes.append(-1)
    # names_list.append('gcratio')

    # if receipt_val['effectiveGasPrice'] > 0 and tr_val['gasPrice']:
    #     list_of_feautes.append(receipt_val['effectiveGasPrice'] / tr_val['gasPrice'])
    # else:
    #     list_of_feautes.append(-1)
    # names_list.append('gas_price_ratio')

    list_of_feautes.append(receipt_val['gasUsed'])
    names_list.append('gasUsed')
    list_of_feautes.append(receipt_val['type'])
    names_list.append('type')
    list_of_feautes.append(len(receipt_val['logs']))
    names_list.append('#logs')
    dd= [len(j['topics']) for j in receipt_val['logs'] ]
    s= len(dd)
    if s>0:
        mm =sum(dd)/s
    else:
        mm=0
    list_of_feautes.append(mm )
    names_list.append('average_topic')
    list_of_feautes.append(tr_val['gas'])
    names_list.append('gas')

    #gas price == effect
    # list_of_feautes.append(ut0.convert_to_gwei*tr_val['gasPrice'])
    # names_list.append('gasPrice')

    if 'maxFeePerGas'in tr_val:
        xx= convert_to_gwei*tr_val['maxFeePerGas']
    else:
        xx= -1
    list_of_feautes.append(xx)
    names_list.append('maxFeePerGas')
    if 'maxPriorityFeePerGas'in tr_val:
        xx= convert_to_gwei*tr_val['maxPriorityFeePerGas']
    else:
        xx= -1
    list_of_feautes.append(xx)
    names_list.append('mxpriorgas')

    if 'maxPriorityFeePerGas' in tr_val and tr_val['maxPriorityFeePerGas'] > 0 and tr_val['maxFeePerGas'] > 0:
        list_of_feautes.append(tr_val['maxPriorityFeePerGas']  / tr_val['maxFeePerGas'])
    else:
        list_of_feautes.append(-1)
    names_list.append('PRIFFEE')


    list_of_feautes.append(tr_val['v'])
    names_list.append('v')
    # list_of_feautes.append(tr_val['r'].hex())
    # names_list.append('r')
    # list_of_feautes.append(tr_val['s'].hex())
    # names_list.append('s')
    list_of_feautes.append(convert_to_gwei*tr_val['value'])
    names_list.append('value')

    all_topics =collect_topics(receipt_val['logs'])
    # list_of_topics= list_of_topics[20:]
    # for j,topic in enumerate(list_of_topics):
    #     names_list.append("tppic_"+str(j))
    #     tt =int (topic in all_topics)
    #     list_of_feautes.append(tt)
    the_input = tr_val['input'].hex()
    a,b,c,d= octet_ob.generate_octet_features(the_input)
    list_of_feautes.extend([a,b,c,d])
    names_list.extend(["len_octet","valid_octet","benign_octet","mal_octet"])
    if save_names:
        with open ('featrue_names.pkl','wb') as file:
            pickle.dump(names_list,file)


    list_of_feautes.append(label)

    np.save( outpfile,np.asarray(list_of_feautes))
    return
def geneate_tensors_for_single_tx(dic_val,rec_ind,tx_ind,label,list_of_topics,outpfile,save_names=False ):
    names_list = []
    list_of_feautes = []
    receipt_val = dic_val[rec_ind]
    tr_val = dic_val[tx_ind]
    gw_effective = convert_to_gwei * receipt_val['effectiveGasPrice']
    list_of_feautes.append(gw_effective)
    names_list.append('effectiveGasPrice_gwei')

    list_of_feautes.append(receipt_val['cumulativeGasUsed'])
    names_list.append('cumulGused')
    if tr_val['gas'] > 0 and receipt_val['gasUsed'] > 0:
        list_of_feautes.append(receipt_val['gasUsed'] / tr_val['gas'])
    else:
        list_of_feautes.append(-1)
    names_list.append('gas_ratio')

    list_of_feautes.append(receipt_val['gasUsed'])
    names_list.append('gasUsed')
    list_of_feautes.append(receipt_val['type'])
    names_list.append('type')
    list_of_feautes.append(len(receipt_val['logs']))
    names_list.append('#logs')
    dd = [len(j['topics']) for j in receipt_val['logs']]
    s = len(dd)
    if s > 0:
        mm = sum(dd) / s
    else:
        mm = 0
    list_of_feautes.append(mm)
    names_list.append('average_topic')
    list_of_feautes.append(tr_val['gas'])
    names_list.append('gas')

    # gas price == effect
    # list_of_feautes.append(ut0.convert_to_gwei*tr_val['gasPrice'])
    # names_list.append('gasPrice')

    if 'maxFeePerGas' in tr_val:
        xx = convert_to_gwei * tr_val['maxFeePerGas']
    else:
        xx = -1
    list_of_feautes.append(xx)
    names_list.append('maxFeePerGas')
    if 'maxPriorityFeePerGas' in tr_val:
        xx = convert_to_gwei * tr_val['maxPriorityFeePerGas']
    else:
        xx = -1
    list_of_feautes.append(xx)
    names_list.append('mxpriorgas')

    if 'maxPriorityFeePerGas' in tr_val and tr_val['maxPriorityFeePerGas'] > 0 and tr_val['maxFeePerGas'] > 0:
        list_of_feautes.append(tr_val['maxPriorityFeePerGas'] / tr_val['maxFeePerGas'])
    else:
        list_of_feautes.append(-1)
    names_list.append('PRIFFEE')

    list_of_feautes.append(tr_val['v'])
    names_list.append('v')

    list_of_feautes.append(convert_to_gwei * tr_val['value'])
    names_list.append('value')

    all_topics = collect_topics(receipt_val['logs'])
    list_of_topics= list_of_topics[:40]
    for j,topic in enumerate(list_of_topics):
        names_list.append("tppic_"+str(j))
        tt =int (topic in all_topics)
        list_of_feautes.append(tt)


    list_of_feautes.append(label)
    xx =torch.as_tensor(list_of_feautes,dtype=torch.float)
    xx =torch.unsqueeze(xx,dim=1)
    torch.save(xx,outpfile)
    return