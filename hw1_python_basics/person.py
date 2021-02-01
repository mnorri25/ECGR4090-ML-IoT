class person:
	def __init__(self, name, age, height):
		self.name = name
		self.age = age
		self.height = height

	def __repr__(self):
		return (self.name + " is " + str(self.age) + " years old and " + str(self.height) + " cm tall.")

# Test Conditions for creating new_person and repr function based examples given in the homework
#new_person = person(name='Joe', age=34, height=184)
#print(new_person)
