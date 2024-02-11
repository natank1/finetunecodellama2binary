
import requests
import codecs
import binascii
import pickle
import base64
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def force_bytes(value):
    if isinstance(value, bytes):
        return value
    elif isinstance(value, memoryview):
        return bytes(value)
    elif isinstance(value, str):
        return bytes(value, 'utf8')
    else:
        raise TypeError("Unsupported type: {0}".format(type(value)))


def force_text(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, bytes):
        return str(value, 'latin1')
    elif isinstance(value, memoryview):
        return str(bytes(value), 'latin1')
    else:
        raise TypeError("Unsupported type: {0}".format(type(value)))


def remove_0x_prefix(value):
    if force_bytes(value).startswith(b'0x'):
        return value[2:]
    return value


def encode_hex(value):
    tt= force_bytes(value)
    mm =codecs.encode(tt, 'hex')
    zz=decode_hex(mm)
    return b'0x' + codecs.encode(tt, 'hex')


def decode_hex(value):
    ll= force_bytes(value)
    codecs.decode(ll,'hex')
    return codecs.decode(remove_0x_prefix(force_bytes(value)), 'hex')

def update_dic(data,dic_hex,dic_byte):
    for i in data:
        bytes_struc= encode_hex(i["text_signature"])
        dic_byte.setdefault(bytes_struc,i["text_signature"] )
        dic_hex.setdefault(i["hex_signature"],(i["text_signature"],i['id']))
    return dic_hex,dic_byte


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dic_hex ={}
    dic_byte= {}
    for j in range (1,10000):
        # url = "https://www.4byte.directory/api/v1/event-signatures/?page="+str(j)
        url = "https://www.4byte.directory/api/v1/signatures/?page=" + str(j)
        response = requests.get(url)
        if response.status_code==200:
            mm = response.json()
            data =mm["results"]
            update_dic(data, dic_hex, dic_byte)
            if j% 100==3:
                print (j, len(dic_byte),len(dic_hex))
                with open("file_namehexa.pkl", "wb") as file:
                     pickle.dump(dic_hex, file)
                with open("file_namebyts.pkl", "wb") as file:
                    pickle.dump(dic_byte, file)
        else:
            print (j)
        a=1

    with open("file_namehexa.pkl", "wb") as file:
        pickle.dump(dic_hex, file)
    with open("file_namebyts.pkl", "wb") as file:
        pickle.dump(dic_byte, file)
    print ("ok it works")
    exit()
    # url = "https://www.4byte.directory/api/v1/signatures/"
#     response = requests.get(url)
#     mm=response.json()
#
#     response = requests.get("https://www.4byte.directory/api/v1/signatures/?format=json" )
#     data =response.json()['results']
#
#     ll=data[0]['hex_signature']
#     resp = requests.get("https://www.4byte.directory/api/v1/signatures/?hex_signature=%s" %ll)
#     encode_hex((data[0]['text_signature']))
#     a=1
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
# # bytes4_signature = make_4byte_signature(self.text_signature)
