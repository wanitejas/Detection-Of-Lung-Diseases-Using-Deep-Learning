from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES

#salt = get_random_bytes(16) # 16 bytes * 8 = 128 bits (1 byte = 8 bits)
salt = b'\x0e\xe9\xadKi\xfd\xcb:\xcd\xc0\x9cV\x072\x00\xe0'

password = input("\nEnter password: ") # Password provided by the user, can use input() to get this

key = PBKDF2(password, salt, dkLen=16) # Your key that you can encrypt with


#######Encryption############
input_file = open("Original XRay.png", "rb") #readbinary
input_data = input_file.read()
input_file.close()

cipher = AES.new(key, AES.MODE_CFB) # CFB mode
ciphered_data = cipher.encrypt(input_data) # Only need to encrypt the data, no padding required for this mode

enc_file = open("Encrypted XRay.png", "wb")
enc_file.write(cipher.iv)
enc_file.write(ciphered_data)
enc_file.close()

print("\n\nFile encrypted...")

a = input('\n\nPress any key to exit')
if a:
    exit(0)




