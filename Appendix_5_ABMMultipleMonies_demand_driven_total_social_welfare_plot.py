import numpy as np
import random
from scipy.stats import gamma
from scipy.stats import skewnorm
import scipy.linalg
from mesa import Agent
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
import matplotlib.pyplot as plt
import math
import pandas

#Money-Supplying Agent Class.
class MoneySupplier(Agent):

	#constructor method.
	def __init__(self, name, model, num_monies, market):
		
		#defining market variable according to the market argument passed to agent.
		self.market = market
		#initialising portfolio of monies held by money-supplier. 
		self.portfolio = np.zeros(num_monies)
		
		self.utility = 0.0
		self.name = name
		self.num_monies = num_monies
		
		#boolean to determine whether the agent is the money-expander or not.
		self.money_expander = False
		#this is the np.arange() array.  
		self.prop_money_to_test = model.prop_money_to_test
		
		
		#random number generated for gamma shape (strictly positive).
		self.gamma_shape = random.uniform(0.01, 10)
		#random number generated for skew normal shape (real).
		self.skewnorm_shape = random.uniform(-5, 5)
		#random number generated for gamma scale (strictly positive).
		self.gamma_scale = random.uniform(0.01, 2)
		#random number generated for skew normal scale (positive, real).
		self.skewnorm_scale = random.uniform(0.01, 2)
		
		#initialising exchange and interest rate preferences held by the money-supplier
		#for their money.
		self.exchange_pref_dist = gamma(self.gamma_shape, scale = self.gamma_scale)
		self.interest_pref_dist = skewnorm(self.skewnorm_shape, scale = self.skewnorm_scale)
		
		self.calc_self_utility()
	
	#method to return preferences of this agent.
	def return_preferences(self):
		return self.exchange_pref_dist, self.interest_pref_dist
		
	def receive_monies(self, list_monies_received):
		self.portfolio += list_monies_received
		
	def give_monies(self, list_monies_given):
		self.portfolio -= list_monies_given
		return list_monies_given
	
	#method to calculate agent's utility based on prevailing interest rates
	#and exchange rates.
	def calc_self_utility(self):
		
		interest_rates, exchange_rates = self.market.return_utility_rates()
		
		#subtract 1 because name variable does not begin with 0 by default for the purposes
		#easing understanding for non-CS folk.
		index = self.name[1] - 1 
		
		utility = self.interest_pref_dist.pdf(interest_rates[index])
		utility += self.exchange_pref_dist.pdf(exchange_rates[index])
		
		self.utility = utility
	
	#method to change the self.money_expander variable to True so that it is known that
	#this agent expanded its money supply. 
	def set_expander_true(self):
		self.money_expander = True
		
	def create_money(self, amount):
		index = self.name[1] - 1
		self.portfolio[index] += amount
		self.market.add_money(index, amount)
		self.market.update_utility_rates()
		self.market.update_money_supplier_utility()
		
	def destroy_money(self, amount):
		index = self.name[1] - 1
		self.portfolio[index] -= amount
		self.market.subtract_money(index, amount)
		self.market.update_utility_rates()
		self.market.update_money_supplier_utility()
	
	#determine optimum money-supply proportion.
	def determine_opt_money_supply(self):
		#initialises opt money supply variable.
		opt_money_supply = 0.0
		
		# -1 from name[1] because they started with +1 to begin with. 
		money_supplier_index = self.name[1] - 1
		
		volumes = self.market.return_volume_of_each_money()
		interest_rates, exchange_rates = self.market.return_utility_rates()
		bilateral_rates = self.market.return_bilateral_rates()
		delta_values = self.market.return_delta_error_values()
		
		temp_volume_of_each_money = np.zeros(self.num_monies)
		temp_default_interest_rates = np.zeros(self.num_monies)
		temp_default_exchange_rates = np.zeros(self.num_monies)
		for i in range(self.num_monies):
			temp_volume_of_each_money[i] = volumes[i]
			temp_default_interest_rates[i] = interest_rates[i]
			temp_default_exchange_rates[i] = exchange_rates[i]
		temp_interest_matrix = np.zeros((self.num_monies, self.num_monies))
		results = np.zeros(len(self.prop_money_to_test))
		
		#current proportion of total money that the money-supplier's money makes up.
		current_prop = (temp_volume_of_each_money[money_supplier_index] / np.sum(temp_volume_of_each_money))
		#the sum of all monies in circulation except the money-supplier's money.
		temp_volume_sum_excluding_supplier = np.sum(temp_volume_of_each_money) - temp_volume_of_each_money[money_supplier_index]
		
		#loop to determine and assign results from numerical testing.
		loop_index = 0
		for x in self.prop_money_to_test:
			#formula to determine the exact quantity of the money that has to be tested 
			#before assigning it to the particular list. 
			test_quantity = (x * temp_volume_sum_excluding_supplier) / (1 - x)
			temp_volume_of_each_money[money_supplier_index] = test_quantity
			temp_sum_of_money = np.sum(temp_volume_of_each_money)
			#setting up interest coefficient matrix to solve the new interest rates. 
			for i in range(self.num_monies):
				for j in range(self.num_monies):
					if i == j:
						temp_interest_matrix[i][j] = 1
					else: 
						temp_interest_matrix[i][j] = -((temp_volume_of_each_money[j])/(temp_sum_of_money))
			temp_default_interest_rates = scipy.linalg.solve(temp_interest_matrix, delta_values)
			
			#determining new, weighted exchange rates for this test. 
			weighted_exchange_rates = []
			for i in range(self.num_monies):
				weighted_exchange_rate = 0.0
				for j in range(self.num_monies):
					#for the fact that it does not weight itself. 
					if i == j:
						weighted_exchange_rate += 0
					else: 
						weighted_exchange_rate += (bilateral_rates[i][j]*(temp_volume_of_each_money[j]/temp_sum_of_money))
				weighted_exchange_rates.append(weighted_exchange_rate)
			for i in range(self.num_monies):
				temp_default_exchange_rates[i] = weighted_exchange_rates[i]
			
			#calculating utility from this test:
			exchange_rate = temp_default_exchange_rates[money_supplier_index]
			interest_rate = temp_default_interest_rates[money_supplier_index]
			utility = self.exchange_pref_dist.pdf(exchange_rate)
			utility += self.interest_pref_dist.pdf(interest_rate)
			results[loop_index] = utility
			loop_index += 1
		
		opt_index = np.argmax(results)
		
		opt_prop = self.prop_money_to_test[opt_index]
		opt_money_supply = (opt_prop * temp_volume_sum_excluding_supplier) / (1 - opt_prop)
		
		return opt_money_supply
	
	def return_utility(self):
		return self.utility
	
	def step(self):
		pass
	
#Money-Demanding Agent Class.
class MoneyDemander(Agent):
	
	#constructor method.
	def __init__(self, name, model, Starting_Money, Number_of_Monies, market):
		self.portfolio = np.zeros(Number_of_Monies)
		self.current_utility = 0.0
		self.Starting_Money = Starting_Money
		self.Number_of_Monies = Number_of_Monies
		self.market = market
		self.name = name
		
		#random number generated for gamma shape (strictly positive)
		self.gamma_shape = random.uniform(0.01, 10)
		#random number generated for skew normal shape (real).
		self.skewnorm_shape = random.uniform(-5, 5)
		#random number generated for gamma scale (strictly positive)
		self.gamma_scale = random.uniform(0.01, 2)
		#random number generated for skew normal scale (positive, real)
		self.skewnorm_scale = random.uniform(0.01, 2)
		
		self.exchange_preference_dist = gamma(self.gamma_shape, scale = self.gamma_scale)
		self.interest_preference_dist = skewnorm(self.skewnorm_shape, scale = self.skewnorm_scale)

		#initialises initial money distribution variable with equal division.
		self.initial_money_distribution = self.Starting_Money / self.Number_of_Monies
		#agent starts with equal amount of each money.
		for i in range(Number_of_Monies):
			self.portfolio[i] = self.initial_money_distribution
		
		self.current_utility = self.calc_Current_Utility()		
	
	#method to calculate current utility.
	def calc_Current_Utility(self):
		current_utility = 0.0
		weighted_interest_rates, weighted_exchange_rates = self.market.return_utility_rates()
		utility_coefficients = []
		
		#calculate the utility associated with each interest rate and each exchange 
		#rate, add them together and then append them to the 'utility coefficients'
		#list predefined within this method.
		for i in range(self.Number_of_Monies):
			interest = weighted_interest_rates[i]
			exchange = weighted_exchange_rates[i]
			interest_utility = self.interest_preference_dist.pdf(interest)
			exchange_utility = self.exchange_preference_dist.pdf(exchange)
			utility = interest_utility + exchange_utility
			utility_coefficients.append(utility)
		
		#adding current utility based on utility coefficients corresponding to each money
		#multiplied by quantity of each money.
		for i in range(self.Number_of_Monies):
			current_utility += utility_coefficients[i]*self.portfolio[i]
		
		return current_utility
	
	#method to return utility coefficients list (for each of the monies). 
	def determine_utility_coefficients(self):
		
		weighted_interest_rates, weighted_exchange_rates = self.market.return_utility_rates()
		utility_coefficients = np.zeros(self.Number_of_Monies)
		
		#calculate the utility associated with each interest rate and each exchange 
		#rate, add them together and then append them to the 'utility coefficients'
		#list predefined within this method.
		for i in range(self.Number_of_Monies):
			interest = weighted_interest_rates[i]
			exchange = weighted_exchange_rates[i]
			interest_utility = self.interest_preference_dist.pdf(interest)
			exchange_utility = self.exchange_preference_dist.pdf(exchange)
			utility = interest_utility + exchange_utility
			utility_coefficients[i] = utility
		
		return utility_coefficients
	
	#method to receive a list of monies from another agent corresponding to each type
	#of money. The positive amount of each money received is credited to the
	#money-demanding agent's portfolio variable.
	def receive_monies(self, monies_received):
		self.portfolio += monies_received
	
	def give_monies(self, monies_given):
		self.portfolio -= monies_given 
	
	#determine and return the optimal quantities of money required for money-demander
	#to maximise utility given the budget constraints. 
	def optimal_portfolio_quantities(self):
		#returns the weighted interest rates, weighted exchange rates and 
		#bilateral exchange rates. The first two are necessary for utility coefficient
		#maximisation and the second is necessary for the budget constraint.
		interest_rates, exchange_rates = self.market.return_utility_rates()
		bilateral_exchange_rates = self.market.return_bilateral_rates()
		#this method uses another method to return the corresponding utility coefficients 
		#for each of the monies respectively.
		utility_coefficients = self.determine_utility_coefficients()
		negative_utility_coefficients = [-x for x in utility_coefficients]
		volume_of_each_money = self.market.volume_of_each_money
		
		#initialising bounds list
		bounds_ub = []
		#initialising coefficient matrix as transpose of bilateral exchange rates matrix.
		#the intuition behind this is that each row would correspond to the constraints 
		#on how many of each money j (from 0 to n - 1) can be bought depending on the 
		#bilateral exchange rates and, thus, the transpose is needed.
		
		#coefficientMatrix = np.matrix.transpose(bilateral_exchange_rates)
		
		#populating the A_upperbound matrix with the transpose of the bilateral exchange
		#rates matrix as well as a diagonal of 1s for the bottom half to represent
		#the fact that, along with the bounds, the total quantity of each money
		#cannot exceed the total volume of that money in circulation (appended to bounds 
		#list later).
		A_upperbound = np.zeros(((self.Number_of_Monies + self.Number_of_Monies),self.Number_of_Monies))
		for i in range(self.Number_of_Monies):
			for j in range(self.Number_of_Monies):
				if i == j:
					A_upperbound[i][j] = 1
					A_upperbound[i+self.Number_of_Monies][j] = 1
				else: 
					A_upperbound[i][j] = bilateral_exchange_rates[j][i]
				
		#iterating through the rows and columns of the bilateral exchange rates to
		#determine the bounds of the inequalities in the constrained optimisation
		#these inequalities are essentially the budget constraint in the sense that
		#each bound represents the maximum possible amount of that money that can be held,
		#given the agent's portfolio and purchasing power. 
		for i in range(self.Number_of_Monies):
			bound = 0.0
			#within one row, iterate through each column.
			for j in range(self.Number_of_Monies):
				#quantity of that money multiplied by the amount of a money which each 
				#unit buys.
				bound += (self.portfolio[j] * bilateral_exchange_rates[j][i])
			
			#the appended bound corresponds to the maximum amount of a particular money
			#that can be bought, given the agent's portfolio of monies and the exchange
			#rates. Each bound corresponds to each money j (from 0 to n-1).
			bounds_ub.append(bound)
		
		#to populate the remaining bounds and upperbound of the matrix. 
		for i in range(self.Number_of_Monies):
			bounds_ub.append(volume_of_each_money[i])
				
		optimal_portfolio = scipy.optimize.linprog(negative_utility_coefficients, b_ub = bounds_ub, A_ub = A_upperbound)
		portfolio_change = np.zeros(self.Number_of_Monies)
		portfolio_change += optimal_portfolio.x
		portfolio_change -= self.portfolio

		quantities_needed = []
		money_indexes = []	
		#method to add the quantities needed of each money to a list as well as the 
		#money indexes of the money-suppliers from which these monies would be sought.
		mon_index = 0
		for x in portfolio_change:
			if x > 0:
				quantities_needed.append(x)
				money_indexes.append(mon_index)
			mon_index += 1
		
		utility_per_change = []
		
		#multiplying the quantities needed of each money by the respective utility
		#coefficients which correspond to the money indexes determined from the previous
		#loop within this method.
		for i in range(len(quantities_needed)):
			utility = quantities_needed[i] * utility_coefficients[money_indexes[i]]
			utility_per_change.append(utility)
		
		#structuring and sorting the utility list - descending by utility gained from
		#each quantity of money so that when it is returned to the money-demanding agent
		#the agent knows which monies to go for (by their importance to it
		#in terms of utility).
		utility_list = [(quantities_needed[i], money_indexes[i], utility_per_change[i]) for i in range(len(quantities_needed))]
		utility_list_sorted = sorted(utility_list, key = lambda x: x[2], reverse = True)
		
		return utility_list_sorted
	
	#method to minimise the utility loss incurred from exchanging one set of monies 
	#for another.
	def minimise_monies_paid(self, monies_demanded):
	
		utility_coefficients = self.determine_utility_coefficients()
		bilateral_rates = self.market.return_bilateral_rates()	
		monies_to_be_paid = []
		
		for x in monies_demanded:
			A_upperbound = np.zeros(((self.Number_of_Monies + 2),self.Number_of_Monies))
			b_bounds = np.zeros(self.Number_of_Monies + 2)
			money_index = x[1]
			quantity_demanded_of_money = x[0]

			#print("quantity demanded of money: ", quantity_demanded_of_money)
			#print("money index: ", money_index)
			#populating the A upper bound matrix with the bilateral exchange rates w.r.t 
			#that particular money. 
			for i in range(self.Number_of_Monies):
				if i == money_index:
					a = quantity_demanded_of_money
					b_bounds[0] = a
					b_bounds[1] = -a
					#because a currency trades itself for one unit each time:
					A_upperbound[0][i] = 1
					A_upperbound[1][i] = -1
				else:
					A_upperbound[0][i] = bilateral_rates[i][money_index]
					A_upperbound[1][i] = -(bilateral_rates[i][money_index])
					#bilateral_rate = bilateral_rates_2[i][money_index]
					#value = quantity_demanded_of_money * bilateral_rate
					#b_equal.append(value)
			
			#populating the upper bound matrix with the constraint of the volume of monies
			#within the agent's portfolio.
			for i in range(self.Number_of_Monies):
				money_bound = self.portfolio[i]
				b_bounds[2+i] = money_bound
				A_upperbound[2+i][i] = 1

			optimal_payment = scipy.optimize.linprog(utility_coefficients, b_ub = b_bounds, A_ub = A_upperbound)
			monies_to_be_paid.append([optimal_payment.x])
		
		return monies_to_be_paid
	
	#same as above method but just for one money rather than multiple monies. 	
	def minimise_money_paid(self, amount_demanded, money_index):
		
		monies_to_be_paid = np.zeros(self.Number_of_Monies)
				
		utility_coefficients = self.determine_utility_coefficients()
		bilateral_rates = self.market.return_bilateral_rates()
		
		A_upperbound = np.zeros(((self.Number_of_Monies + 2),self.Number_of_Monies))
		b_bounds = np.zeros(self.Number_of_Monies+2)
		quantity_demanded_of_money = amount_demanded
		money_index = money_index
		
		if quantity_demanded_of_money > 0:
			for i in range(self.Number_of_Monies):
				if i == money_index:
					a = quantity_demanded_of_money
					b_bounds[0] = a
					b_bounds[1] = -a
					#because a currency buys one of itself each time.
					A_upperbound[0][i] = 1
					A_upperbound[1][i] = -1
				else:
					A_upperbound[0][i] = bilateral_rates[i][money_index]
					A_upperbound[1][i] = -(bilateral_rates[i][money_index])
		
			for i in range(self.Number_of_Monies):
				money_bound = self.portfolio[i]
				b_bounds[2+i] = money_bound
				A_upperbound[2+i][i] = 1
		
			optimal_payment = scipy.optimize.linprog(utility_coefficients, b_ub = b_bounds, A_ub = A_upperbound)
			monies_to_be_paid = optimal_payment.x
		
		
		return monies_to_be_paid
	
	#method to update agent's self.current_utility variable.
	def update_utility(self):			
		self.current_utility = self.calc_Current_Utility()
	
	#method to return agent's utility.
	def return_utility(self):
		return self.current_utility
	
	#step method for money-demanding agent.
	def step(self):
		monies_demanded_with_utilities = self.optimal_portfolio_quantities()
		print(self.name)
		#approach each of the money-suppliers in turn.
		for x in monies_demanded_with_utilities:
			money_index = x[1]
			quantity_demanded = x[0]
			quantity_held_by_supplier = self.market.money_suppliers[money_index].portfolio[money_index]
			
			#if the money-supplier has the money available then they will simply trade
			#with the money-demander and satisfy their demand.
			if quantity_demanded < quantity_held_by_supplier:
				money_receive_list = np.zeros(self.Number_of_Monies)
				money_receive_list[money_index] += quantity_demanded
				optimal_payment = self.minimise_money_paid(quantity_demanded, money_index)
				self.market.money_suppliers[money_index].give_monies(money_receive_list)
				self.market.money_suppliers[money_index].receive_monies(optimal_payment)
				self.receive_monies(money_receive_list)
				self.give_monies(optimal_payment)
			
			#if the money-supplier 
			if quantity_demanded > quantity_held_by_supplier and quantity_held_by_supplier >= 0:
				opt_money_supply = self.market.money_suppliers[money_index].determine_opt_money_supply()
				volume_in_circulation = self.market.volume_of_each_money[money_index]
				diff_with_opt = opt_money_supply - volume_in_circulation
				excess_demand = quantity_demanded - quantity_held_by_supplier
				money_receive_list = np.zeros(self.Number_of_Monies)
				#if it is rational for the supplier to create more money, he does so 
				#and supplies either the amount of excess demand or the money created
				#to the money-demander - excess demand may or may not be fully satisfied. 
				if diff_with_opt > 0:
					two_possibilities = [excess_demand, diff_with_opt]
					money_amount_to_create = min(two_possibilities)
					money_receive_list = np.zeros(self.Number_of_Monies)
					money_amount_to_receive = money_amount_to_create + quantity_held_by_supplier
					money_receive_list[money_index] += money_amount_to_receive
					optimal_payment = self.minimise_money_paid(money_amount_to_receive, money_index)
					self.market.money_suppliers[money_index].create_money(money_amount_to_create)
					self.market.money_suppliers[money_index].give_monies(money_receive_list)
					self.market.money_suppliers[money_index].receive_monies(optimal_payment)
					self.give_monies(optimal_payment)
					self.receive_monies(money_receive_list)
				#if it is not rational for the supplier to create more money, then they
				#simply trade the money they currently have with the demander.
				else: 
					optimal_payment = self.minimise_money_paid(quantity_held_by_supplier, money_index)
					money_receive_list[money_index] += quantity_held_by_supplier
					self.market.money_suppliers[money_index].give_monies(money_receive_list)
					self.market.money_suppliers[money_index].receive_monies(optimal_payment)
					self.give_monies(optimal_payment)
					self.receive_monies(money_receive_list)
			
			
		#after each trade/money-creation, update the utility rates. 
		self.market.update_utility_rates()
		#after each trade/money-creation, update the list of suppliers' utilities.
		self.market.update_money_supplier_utility()
		#after each money-demanding agent's actions, update all money-demanders' utilities.
		self.market.update_money_demander_utility()
				
#Agent Class to hold information of interest rates and exchange rates, with which 
#other agents interact to determine whether monies are sold and bought and which also
#determines whether new monies are created as well as their subsequent prices etc.
class GrandExchange(Agent):

	#instantiates array of the weighted-average, multi-lateral interest rates and 
	#exchange rates which is designed to hold these values corresponding to all the monies
	#that are randomly generated.
	def __init__(self, name, model, number_of_monies, start_volume_each_money, money_suppliers):
		self.number_monies = number_of_monies
		self.default_interest_rates = np.zeros(self.number_monies)
		self.default_exchange_rates = np.zeros(self.number_monies)
		self.number_demanders = model.num_demanders
		self.volume_of_each_money = np.zeros(self.number_monies)
		self.start_volume_each_money = start_volume_each_money
		self.bilateral_exchange_rates = np.zeros((self.number_monies,self.number_monies))
		self.delta_error_interest_values = np.zeros(self.number_monies)
		self.money_suppliers = money_suppliers
		self.money_demanders = model.money_demanders
		
		#adding delta values to the delta/error array/list for use when determining
		#equilibrium interest rates according to the updated interest-parity condition.
		for i in range(self.number_monies):
			delta = random.uniform(-5,5)
			self.delta_error_interest_values[i] = delta
		#sorting delta values in ascending order.
		self.delta_error_interest_values = sorted(self.delta_error_interest_values)
				
		#assigns initial volumes of each type of money within the model/market/system.
		for i in range(self.number_monies):		
			self.volume_of_each_money[i] += self.start_volume_each_money
		
		#initialises interest coefficient matrix for solving.
		interest_coefficient_matrix = np.zeros((self.number_monies, self.number_monies))
		for i in range(self.number_monies):
			sum_of_money = np.sum(self.volume_of_each_money)
			#assigns values to interest coefficient matrix according to proportions of money
			#volume per money.
			for j in range(self.number_monies):
				#ensures diagonal of matrix is all 1's.
				if i == j:
					interest_coefficient_matrix[i][j] = 1
				else:
					interest_coefficient_matrix[i][j] = -((self.volume_of_each_money[j])/(sum_of_money))
		
		#solving for interest rates through scipy linear algebra package.
		default_interest_rates = scipy.linalg.solve(interest_coefficient_matrix, self.delta_error_interest_values)
		for i in range(self.number_monies):
			self.default_interest_rates[i] = default_interest_rates[i]
			
		#initialising/randomising bilateral exchange rates. Row-by-row.
		#each value signifies how much of money j one unit of money i can buy.
		for i in range(self.number_monies):
			#i is the i'th row from 0 up to and including (number_monies - 1).
			for j in range(self.number_monies):
				if i == j:
					self.bilateral_exchange_rates[i][j] = 1
				if j < i:
					self.bilateral_exchange_rates[i][j] = random.uniform(0.2, 40)
		
		#calculating the inverse bilateral exchange rates for the remainder of the matrix
		#based on the aforementioned, predefined bilateral exchange rates.
		for i in range(self.number_monies):
			for j in range(self.number_monies):
				if j > i:
					self.bilateral_exchange_rates[i][j] = (1 / self.bilateral_exchange_rates[j][i])			
		
		#loops to initialise the money-volume-weighted-average exchange rate for each money.
		weighted_exchange_rates = []
		for i in range(self.number_monies):
			weighted_exchange_rate = 0.0
			sum_of_money = np.sum(self.volume_of_each_money)
			for j in range(self.number_monies):
				if i == j:
					weighted_exchange_rate += 0
				else:
					weighted_exchange_rate += (self.bilateral_exchange_rates[i][j]*(self.volume_of_each_money[j]/sum_of_money))
			#weighted_exchange_rate = (weighted_exchange_rate)/(self.number_monies)
			weighted_exchange_rates.append(weighted_exchange_rate)
		
		for i in range(self.number_monies):
			self.default_exchange_rates[i] = weighted_exchange_rates[i]
				
	#returns the list/array of aforementioned interest rates and exchange rates
	#respectively.
	def return_utility_rates(self):
		return self.default_interest_rates, self.default_exchange_rates
	
	#returns list/array of bilateral exchange rates.
	def return_bilateral_rates(self):
		return self.bilateral_exchange_rates
	
	#returns the array of money suppliers for use by the money-demander agents. 
	def return_money_suppliers(self):
		return self.money_suppliers
		#print("number of money suppliers: ", len(self.money_suppliers))
	
	#returns the array of money-demanders.
	def return_money_demanders(self):
		return self.money_demanders
	
	#method for calculating the utility of the money suppliers based upon the weighted
	#exchange rates and interest rates.
	def calc_all_money_suppliers_utility(self):
		
		#initialise money suppliers' utility list.
		money_suppliers_utility = []
		#iterate through money suppliers to retrieve utility from each rate.
		for i in range(self.number_monies):
			exchange_pref, interest_pref = self.money_suppliers[i].return_preferences()
			
			this_supplier_utility = exchange_pref.pdf(self.default_exchange_rates[i])
			this_supplier_utility += interest_pref.pdf(self.default_interest_rates[i])
			
			money_suppliers_utility.append(this_supplier_utility)
		
		#print("all money suppliers utility: ", money_suppliers_utility)
		return money_suppliers_utility
	
	def update_utility_rates(self):
		#loop to update the money-volume weighted-average exchange rates array.
		weighted_exchange_rates = []
		sum_of_money = np.sum(self.volume_of_each_money)
		for i in range(self.number_monies):
			weighted_exchange_rate = 0.0
			for j in range(self.number_monies):
				if i == j:
					weighted_exchange_rate += 0
				else:
					weighted_exchange_rate += (self.bilateral_exchange_rates[i][j]*(self.volume_of_each_money[j]/sum_of_money))
			#weighted_exchange_rate = (weighted_exchange_rate)/(self.number_monies)
			weighted_exchange_rates.append(weighted_exchange_rate)
		#updates agent's self exchange rates variable with the newly calculated ones from 
		#within this method.
		for i in range(self.number_monies):
			self.default_exchange_rates[i] = weighted_exchange_rates[i]
		
		#loops to update the money-volume weighted-average interest rates array according
		#to the interest-parity condition. 
		#initialises interest coefficient matrix for solving.
		interest_coefficient_matrix = np.zeros((self.number_monies, self.number_monies))
		sum_of_money = np.sum(self.volume_of_each_money)
		for i in range(self.number_monies):
			#assigns values to interest coefficient matrix according to proportions of money
			#volume per money.
			for j in range(self.number_monies):
				#ensures diagonal of matrix is all 1's.
				if i == j:
					interest_coefficient_matrix[i][j] = 1
				else:
					interest_coefficient_matrix[i][j] = -((self.volume_of_each_money[j])/(sum_of_money))
		
		#solving for interest rates through scipy linear algebra package.
		default_interest_rates = scipy.linalg.solve(interest_coefficient_matrix, self.delta_error_interest_values)
		#assigning to the self variable for interest rates.
		for i in range(self.number_monies):
			self.default_interest_rates[i] = default_interest_rates[i]
		
	#method to update the utility of the money-suppliers stored in the self variable.
	def update_money_supplier_utility(self):
		#loop to update the utility variable within each supplier agent.
		for i in range(self.number_monies):
			self.money_suppliers[i].calc_self_utility()
	
	#updating all the money-demanders' utility.
	def update_money_demander_utility(self):
		for i in range(self.number_demanders):
			self.money_demanders[i].update_utility()
			
	
	#method to return a list that contains the volume of each money in the system. 
	def return_volume_of_each_money(self):
		return self.volume_of_each_money
		
	def add_money(self, money_index, amount_of_money):
		 self.volume_of_each_money[money_index] += amount_of_money
	
	def subtract_money(self, money_index, amount_of_money):
		self.volume_of_each_money[money_index] -= amount_of_money
	
	def return_delta_error_values(self):
		return self.delta_error_interest_values
		
	#method to deploy helicopter money to test the monetary autonomy /
	#sovereign monetary policy aspect of the trilemma to see if multiple monies should
	#be able to, in theory, solve this aspect of the trilemma. 
	def helicopter_money(self, proportion_total_money, prop_money_expanders, proportion_receivers):
		
		#this is the proportion of the total money supply that will correspond to
		#the total volume of arbitrary money expansion (i.e monetary autonomy, sovereign
		#monetary policy, 'helicopter money' etc.)
		prop_tot_money = proportion_total_money
		#number of money-suppliers that engage in this 'helicopter money'
		prop_agent_expand = prop_money_expanders
		#the proportion of agents that will receive from this money expansion.
		prop_receive = proportion_receivers
		
		#determining the number of agents that expand their respective money supplies.
		num_agent_expanders = prop_agent_expand * self.number_monies
		num_agent_expanders = math.ceil(num_agent_expanders)

		
		sum_of_money = np.sum(self.volume_of_each_money)
		
		helicopter_money_volume_total = prop_tot_money * sum_of_money
		helicopter_money_volume_each = helicopter_money_volume_total / num_agent_expanders
		
		#initialising and defining expander agents' indexes by randomly sampling
		#without replacement. 
		expanders_indexes = random.sample(range(self.number_monies), num_agent_expanders)
		
		#adds the volume of each respective money to the self variable of the market agent.
		for i in expanders_indexes:
			self.volume_of_each_money[i] += helicopter_money_volume_each
		
		#defining and deciding the number of money-receiving agents 
		#based on aforementioned parameters. 
		num_receivers = prop_receive * self.number_demanders
		num_receivers = round(num_receivers)
		#defining monies for each agent. 
		monies_for_each_agent = np.zeros(self.number_monies)
		volume_each_money_for_each_agent = helicopter_money_volume_each / num_receivers	
		
		#for loop to assign values to the predefined monies for each agent list.
		#also makes it so that the money suppliers are marked in that way through 
		#their respective boolean variables.
		for i in expanders_indexes:
			monies_for_each_agent[i] += volume_each_money_for_each_agent
			self.money_suppliers[i].set_expander_true()
		
		#assigning the monies to each agent that is randomly picked without replacement 
		#(i.e the same agent is not assigned monies more than once). 
		receivers_indexes = random.sample(range(self.number_demanders), num_receivers)
		for i in receivers_indexes:
			self.money_demanders[i].receive_monies(monies_for_each_agent)
		
		self.update_utility_rates()

def return_success_rate(model):
	return model.percentage_success
	
def return_money_value_growth_rate(model):
	return model.weighted_money_growth_percentage
	
def return_social_welfare_growth_percentage(model):
	return model.social_welfare_percentage_growth
	
def return_total_social_welfare(model):
	return model.social_welfare

#Model class.
class MyModel(Model):

	#constructor method.
	def __init__(self, NSuppliers, NDemanders, InitialMoney, prop_hel, prop_expand, prop_rec):
		
		#defines variables: number of money-demanders, number of money-suppliers, 
		#initial volume of money in circulation, the money each agent gets initially,
		#the list of money suppliers, the money suppliers' utility
		self.num_demanders = NDemanders
		self.num_suppliers = NSuppliers
		self.initial_money = InitialMoney
		self.each_agent_initial_money = (self.initial_money) / (self.num_demanders)
		self.each_money_initial_volume = (self.initial_money) / (self.num_suppliers)
		self.money_suppliers = []
		self.money_suppliers_utility = np.zeros(self.num_suppliers)
		self.money_demanders = []
		#in format: start (including this value), stop (not including this value), step.
		#this list is for use by the money-supply agents to do numerical approximation. 
		self.prop_money_to_test = np.arange(0.001, 1, 0.001)
		self.prop_hel, self.prop_expand, self.prop_rec = prop_hel, prop_expand, prop_rec
		
		#makes schedule RandomActivation
		self.schedule = RandomActivation(self)
		
		#Generates the GrandExchange Agent for Market with name.
		Market = GrandExchange("Market", self, self.num_suppliers, self.each_money_initial_volume, self.money_suppliers)
		self.market = Market
		#generate and assign names to money-demanding agents + add them to schedule.
		for i in range (self.num_demanders):
			num = i+1
			name = "D",num
			a = MoneyDemander(name, self, self.each_agent_initial_money, self.num_suppliers, Market)
			self.schedule.add(a)
			self.money_demanders.append(a)
		
		#generate and assign names to money-supplying agents. Added to schedule
		#and to self.money_suppliers list to be passed to the Market Agent.
		for i in range(self.num_suppliers):
			num = i+1
			name = "S",num
			a = MoneySupplier(name, self, self.num_suppliers, Market)
			#this simultaneously appends both the self.money_suppliers array at the model
			#level as well as in the Market Agent level.
			self.money_suppliers.append(a)
		
		money_suppliers_utility = Market.calc_all_money_suppliers_utility()
		for i in range(self.num_suppliers):
			self.money_suppliers_utility[i] = money_suppliers_utility[i]
		
		#defining variables to be tested and recorded in each step. 
		self.model_step = 0
		self.initial_exchange_rates = np.zeros(self.num_suppliers)
		self.initial_volume_of_each_money = np.zeros(self.num_suppliers)
		self.initial_exchange_weighted_money_value = 0.0
		self.Helicopter_Money = False
		self.percentage_changes = np.zeros(self.num_suppliers)
		self.percentage_changes_volumes = np.zeros(self.num_suppliers)
		self.final_exchange_weighted_money_value = 0.0
		self.weighted_money_growth_percentage = 0.0
		self.success = 0.0
		self.percentage_success = 0.0
		self.money_demanders_welfare = 0.0
		self.money_suppliers_welfare = 0.0
		self.social_welfare = 0.0
		self.money_demanders_welfare_percentage_growth = 0.0
		self.money_suppliers_welfare_percentage_growth = 0.0
		self.social_welfare_percentage_growth = 0.0
		self.helicopter_success_rate = 0.0
		self.initial_total_money_volume = self.initial_money
		self.final_total_money_volume = 0.0
		
		self.data_collector = DataCollector(
			model_reporters = {"Absolute Social Welfare": return_total_social_welfare})
	
	#defining a step within the model.
	def step(self):
		#setting and storing the initial exchange rates and initial volumes of each money.
		#these will not be automatically updated once the variables in the market change
		#due to actions of agents. This is necessary for measuring the percentage
		#and absolute differences in the system at the start and at the end of each step. 
		for i in range(self.num_suppliers):
			self.initial_exchange_rates[i] = self.market.default_exchange_rates[i]
			self.initial_volume_of_each_money[i] = self.market.volume_of_each_money[i]
		
		#calculating the total utility of all money-supplier agents.
		self.money_suppliers_welfare = 0.0
		for i in range(self.num_suppliers):
			supplier_utility = self.market.money_suppliers[i].return_utility()
			self.money_suppliers_welfare += supplier_utility
		
		#calculating the total utility of all money-demanding agents. 
		self.money_demanders_welfare = 0.0
		for i in range(self.num_demanders):
			demander_utility = self.market.money_demanders[i].return_utility()
			self.money_demanders_welfare += demander_utility
		
		self.social_welfare = self.money_suppliers_welfare + self.money_demanders_welfare
		
		print("INITIAL EXCHANGE RATES: ", self.initial_exchange_rates)
		print()
		print("INITIAL VOLUME OF EACH MONEY: ", self.initial_volume_of_each_money)
		print()
		
		self.initial_exchange_weighted_money_value = 0.0
		for i in range(self.num_suppliers):
			self.initial_exchange_weighted_money_value += (self.initial_exchange_rates[i] * self.initial_volume_of_each_money[i])
		
		#This resets the helicopter money variable as being false to ensure that it is 
		#known that helicopter money is not implemented in this step unless the
		#self.model_step variable is equal to 29 (which is basically the 30th step). 
		self.Helicopter_Money = False
		if self.model_step == 29:
			self.market.helicopter_money(self.prop_hel, self.prop_expand, self.prop_rec)
			self.Helicopter_Money = True
		
		initial_suppliers_welfare = self.money_suppliers_welfare
		initial_demanders_welfare = self.money_demanders_welfare
		initial_social_welfare = self.social_welfare
		
		self.schedule.step()
		print("VOLUME OF EACH MONEY AFTER STEP: ", self.market.volume_of_each_money)
		print()
		print("INTEREST RATES: ", self.market.default_interest_rates)
		print()
		print("EXCHANGE RATES: ", self.market.default_exchange_rates)
		print()
		print("Percentage Changes of Exchange Rates: ")
		
		index = 0
		for x in self.initial_exchange_rates:
			ratio = (self.market.default_exchange_rates[index]/x)
			percentage_change = ratio - 1
			percentage_change = (percentage_change * 100)
			self.percentage_changes[index] = percentage_change
			index += 1
		
		print(self.percentage_changes)
		print()
		
		index = 0
		for x in self.initial_volume_of_each_money:
			ratio = (self.market.volume_of_each_money[index]/x)
			percentage_change = ratio - 1
			percentage_change = (percentage_change * 100)
			self.percentage_changes_volumes[index] = percentage_change
			index += 1
		print("PERCENTAGE CHANGES VOLUMES OF EACH MONEY: ")
		print(self.percentage_changes_volumes)
		print()
		
		self.final_exchange_weighted_money_value = 0
		for i in range(self.num_suppliers):
			self.final_exchange_weighted_money_value += (self.market.default_exchange_rates[i] * self.market.volume_of_each_money[i])
		
		weighted_money_growth_ratio = (self.final_exchange_weighted_money_value / self.initial_exchange_weighted_money_value)
		self.weighted_money_growth_percentage = (weighted_money_growth_ratio - 1)
		self.weighted_money_growth_percentage = (self.weighted_money_growth_percentage * 100)
		print("Weighted Money Value Growth Percentage: ", self.weighted_money_growth_percentage, "%")
		print()
		
		self.success = False
		num_success = 0.0
		for x in self.percentage_changes:
			if x < 1 and x > -1:
				num_success += 1

		if num_success >= 1:
			self.success = True
			
		#re-calculating the total utility of all money-supplier agents.
		self.money_suppliers_welfare = 0.0
		for i in range(self.num_suppliers):
			supplier_utility = self.money_suppliers[i].return_utility()
			self.money_suppliers_welfare += supplier_utility
		
		#re-calculating the total utility of all money-demanding agents. 
		self.money_demanders_welfare = 0.0
		for i in range(self.num_demanders):
			demander_utility = self.money_demanders[i].return_utility()
			self.money_demanders_welfare += demander_utility
		
		self.social_welfare = self.money_suppliers_welfare + self.money_demanders_welfare
		
		print("INITIAL SOCIAL WELFARE: ", initial_social_welfare)
		print("INITIAL MONEY DEMANDER WELFARE: ", initial_demanders_welfare)
		print("INITIAL MONEY SUPPLIERS WELFARE: ", initial_suppliers_welfare)
		print()
		percentage_change_social_welfare = (self.social_welfare / initial_social_welfare)
		percentage_change_social_welfare = percentage_change_social_welfare - 1
		percentage_change_social_welfare = percentage_change_social_welfare * 100
		self.social_welfare_percentage_growth = percentage_change_social_welfare
		
		percentage_change_suppliers_welfare = (self.money_suppliers_welfare / initial_suppliers_welfare)
		percentage_change_suppliers_welfare = percentage_change_suppliers_welfare - 1
		percentage_change_suppliers_welfare = percentage_change_suppliers_welfare * 100
		self.money_suppliers_welfare_percentage_growth = percentage_change_suppliers_welfare
		
		percentage_change_demanders_welfare = (self.money_demanders_welfare / initial_demanders_welfare)
		percentage_change_demanders_welfare = percentage_change_demanders_welfare - 1
		percentage_change_demanders_welfare = percentage_change_demanders_welfare * 100
		self.money_demanders_welfare_percentage_growth = percentage_change_demanders_welfare
		
		print("FINAL SOCIAL WELFARE: ", self.social_welfare)
		print("SOCIAL WELFARE PERCENTAGE GROWTH: ", self.social_welfare_percentage_growth)
		print()
		print("FINAL MONEY SUPPLIERS WELFARE:", self.money_suppliers_welfare)
		print("MONEY SUPPLIERS WELFARE PERCENTAGE GROWTH: ", self.money_suppliers_welfare_percentage_growth)
		print()
		print("FINAL MONEY DEMANDERS WELFARE: ", self.money_demanders_welfare)
		print("MONEY DEMANDERS WELFARE PERCENTAGE GROWTH: ", self.money_demanders_welfare_percentage_growth)
		
		self.percentage_success = (num_success/len(self.percentage_changes))*100
		if self.model_step == 29:
			self.helicopter_success_rate = self.percentage_success
		print()
		
		print("HELICOPTER MONEY: ", self.Helicopter_Money)
		print()
		
		print("TEST SUCCESS: ", self.success)
		print()
		
		print("PERCENTAGE SUCCESS: ", self.percentage_success, "%")
		print()
		
		print("=========== MODEL STEP DONE ==========", self.model_step)
		self.data_collector.collect(self)
		self.model_step += 1

#instantiates model.
model = MyModel(10, 10, 1000, 0.1, 0.1, 0.5)
#runs multiple steps of the model
for i in range(50):
	model.step()
total_social_welfare = model.data_collector.get_model_vars_dataframe()
total_social_welfare.plot()
plt.show()