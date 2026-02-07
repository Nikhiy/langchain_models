from pydantic import BaseModel,EmailStr,Field# this emailstr checks if it correct email if not it wont accpt the email and throws error
from typing import Optional
class Student(BaseModel):
    name:str
    age:Optional[int]=None #optional field
    fav_number:Optional[int]=None
    email:EmailStr
    cgpa:Field(gt=0,lt=10)#this makes sure the input is between this range
new_student={'name':'nikhil','age':5,'fav_number':'32','email':'qwert@gmail.com'}#even if we give a string as input for fav_numer where we should give number it will just conver it to int
student=Student(**new_student)
print(student)

#now check how we will send this new student to the model in "with_structured_output_typeddict.py" file