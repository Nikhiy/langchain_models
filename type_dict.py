from typing import TypedDict

class Person(TypedDict):
    nme:str
    age:int

new_preson:Person={'name':'nikhil','age':18}
print(new_preson)