# Lingorithm Text Classification
This package is the core function for any NLP operation or pacakge used by lingorithm.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install lim.

```bash
pip install lim
```

# Usage
### Data 
Input data is a list of tuples 
```Python
data = [
    ("Help me please", "commonQ.assist"),
    ("Is this a bot?", "commonQ.bot"),
    ("Is there a bot chatting to me?", "commonQ.bot")
]
```
```Python
import ltc

tc = ltc('en', data) 

tc.train(epochs=20, batch_size=8, model_name="commonQ_v1")


classified = tc.classify("Help me please")

print(classified)
# [('commonQ.assist', 0.9737932),
#  ('commonQ.not_giving', 0.012769885),
#  ('commonQ.name', 0.0071987235),
#  ('commonQ.how', 0.002869619),
#  ('commonQ.query', 0.0023981368),
#  ('commonQ.bot', 0.000478917),
#  ('faq.bad_service', 0.0003706195),
#  ('faq.aadhaar_missing', 5.4887507e-05),
#  ('faq.apply_register', 3.3203887e-05),
#  ('commonQ.just_details', 1.60179e-05),
#  ('faq.biz_new', 9.965993e-06),
#  ('faq.application_process', 3.101246e-06),
#  ('faq.borrow_limit', 1.6019511e-06),
#  ('faq.borrow_use', 1.4953373e-06),
#  ('commonQ.wait', 2.9794015e-07),
#  ('faq.biz_simpler', 2.6959242e-07),
#  ('contact.contact', 9.587218e-08),
#  ('faq.biz_category_missing', 5.7328425e-08),
#  ('faq.approval_time', 5.5687766e-09),
#  ('faq.banking_option_missing', 1.6084242e-09),
#  ('faq.address_proof', 1.3430994e-09)]

```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)