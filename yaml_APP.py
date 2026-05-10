import streamlit_authenticator as stauth
import yaml

users =                          ["Admin"    , "ybouhlal"       , "aamraoui"     , "bbennis"    , "guest"]
names =                          ["Admin"    , "Bouhlal Yassine", "Amraoui Amine", "Bennis Badr", "guest"]
hashed_passwords = stauth.Hasher(["Admin_mdp", "yb_mdp"         , "aAmr.9054!7"  , "dra.5847"   , "guest"]).generate()

# Open the YAML file and load the config dictionary
with open("files/hash_APP.yaml") as file :
    config = yaml.load(file, Loader=yaml.SafeLoader)

# Replace the plain text passwords with the hashed passwords
for i in range(len(users)):
    config['credentials']['usernames'][users[i]]['password'] = hashed_passwords[i]
    config['credentials']['usernames'][users[i]]['name'] = names[i]

# Open the YAML file again and write the updated config dictionary
with open("files/hash_APP.yaml", 'w') as file:
    yaml.dump(config, file)