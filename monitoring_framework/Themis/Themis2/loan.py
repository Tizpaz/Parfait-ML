

def loan(vars):
	sex = vars[0]
	race = vars[1]
	income = vars[2]
	if sex == "male":
		return True
	elif race != "green":
		if income == "0...50000":
			return False
		else:
			return True
	else:
		if income == "0...50000" or income == "50001...100000":
			return False
		else:
			return True



