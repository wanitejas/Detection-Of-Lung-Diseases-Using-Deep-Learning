from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES

#salt = get_random_bytes(16) # 16 bytes * 8 = 128 bits (1 byte = 8 bits)
salt = b'\x0e\xe9\xadKi\xfd\xcb:\xcd\xc0\x9cV\x072\x00\xe0'

password = input("\nEnter password: ") # Password provided by the user, can use input() to get this

key = PBKDF2(password, salt, dkLen=16) # Your key that you can encrypt with


#######Decryption############
enc_file = open("Encrypted XRay.png", "rb")
iv = enc_file.read(16)
enc_data = enc_file.read()
enc_file.close()

cfb_decipher = AES.new(key, AES.MODE_CFB, iv=iv)
plain_data = cfb_decipher.decrypt(enc_data)

output_file = open("Decrypted XRay.png", "wb")
output_file.write(plain_data)
output_file.close()

print("\n\nFile decrypted...")

a = input('\n\nPress any key to exit')
if a:
    exit(0)


