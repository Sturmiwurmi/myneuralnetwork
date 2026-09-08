extends Node2D

@export var learningRate = 0.05
#dataset
@export var loadMax = 100
@export var numbersLoaded = 0
@export var path:String
@export var delay:float
var dataset
#datasetend

const NEURON_SZENE = preload("res://Scenes/neuron.tscn")
@export var LAYERS:Array = []
@onready var pixelgrid:Pixelgrid = $Pixelgrid
@onready var GuessLabel:Label = $Guesslabel


func _ready() -> void: 
	_createLayers() 
	_connectLayers()
	
	
	for i in range(LAYERS.size()):
		for k in range(LAYERS[i].size()):
			LAYERS[i][k].initWeigths()
	pass 
	print("Loading Dataset")
	dataset = load_mnist_database(path)
	print("DONE")
	
	TRAIN1(1000)
	while(true):
		for i in range(dataset.size()):
			cost(i)
			await get_tree().create_timer(delay).timeout

func TRAIN1(SequenceLimiter)->void:
	var file = FileAccess.open("res://TRAIN1.txt", FileAccess.WRITE)
	
	print("Starting Training 1")
	var n = 100.0
	#first n trainingsequenzes are reserved to measure network
	var precision = 0
	for i in range(n):
		if(guess(i)):
			precision+=1
	precision = precision/n
	print("Guessed Correctly Without Training "+str(precision*100)+"%")
	file.store_string("Guessed Correctly Without Training "+str(precision*100)+"%\n")
	print("Cost of first Example "+ str(cost(0)))
	file.store_string("Cost of first Example "+ str(cost(0))+"\n")
	
	
	for i in range(n,SequenceLimiter + n):
		backProp(i)
		applyNewWeights()
		if i % 100 == 0:
			var fortschritt = (float(i - n) / float(SequenceLimiter)) * 100.0
			print("Training: ", fortschritt, "%")
		#print("random weight"+str(LAYERS[2][4].weigths[2])) #random weight to check its not exploding
	pass
	
	file.store_string("Trained on "+str(SequenceLimiter)+" sequences"+"\n")
	print("Trained on "+str(SequenceLimiter)+" sequences")

	precision = 0
	for i in range(n):
		if(guess(i)):
			precision+=1
	precision = precision/n
	print("Guessed Correctly With Training "+str(precision*100)+"%")
	file.store_string("Guessed Correctly With Training "+str(precision*100)+"%\n")
	print("Cost of first Example "+ str(cost(0)))
	file.store_string("Cost of first Example "+ str(cost(0))+"\n")
	pass
	
func applyNewWeights() -> void:
	for i in range(LAYERS.size()):
		for j in range(LAYERS[i].size()):
			LAYERS[i][j].applyWeightWishes()
	pass
	
func backProp(dataSetIndex) -> void:
	pixelgrid.drawNumber(getData(dataSetIndex))
	var currInput = getData(dataSetIndex)
	for i in range(currInput.size()):
		currInput[i] = currInput[i] / 255.0
	var expectedTestNumber = getExpectedNumber(dataSetIndex)
	
	# --- VORWÄRTSPASS (Unverändert) ---
	for layer in range(LAYERS.size()):
		var tmpInput = []
		for neuron in range(LAYERS[layer].size()):
			LAYERS[layer][neuron].setIndexInLayer(neuron)
			tmpInput.append(LAYERS[layer][neuron].activate(currInput))
		currInput = tmpInput
		
	# --- RÜCKWÄRTSPASS ---
	
	# Wir speichern jetzt nicht mehr die Gewichte für die vorherige Schicht,
	# sondern die BERECHNETEN FEHLER (Deltas) der aktuellen Schicht.
	var fehler_der_aktuellen_schicht = [] 
	
	for currLayer in range(LAYERS.size()-1, 0, -1):
		
		var fehler_fuer_naechste_schicht_rueckwaerts = [] # Das wird der Fehler für currLayer - 1
		# Bereite das Array für die vorherige Schicht vor (mit Nullen füllen)
		fehler_fuer_naechste_schicht_rueckwaerts.resize(LAYERS[currLayer-1].size())
		fehler_fuer_naechste_schicht_rueckwaerts.fill(0.0)

		for index in range(LAYERS[currLayer].size()):
			var currNeuron:Neuron = LAYERS[currLayer][index]
			var currOutput = currNeuron.currOutput
			
			var fehler_dieses_neurons = 0.0
			
			# 1. FEHLER BERECHNEN
			if(currLayer == LAYERS.size()-1): 
				# Letzte Schicht: Fehler ist (Ziel - Output)
				var ziel = 0.0
				if index == expectedTestNumber:
					ziel = 1.0
				fehler_dieses_neurons = ziel - currOutput
			else:
				# Versteckte Schicht: Fehler kommt aus dem Array, das im vorherigen Schleifendurchlauf gefüllt wurde
				fehler_dieses_neurons = fehler_der_aktuellen_schicht[index]
			
			
			# WICHTIG: Die Ableitung der Sigmoid-Funktion!
			var gradient = fehler_dieses_neurons * (currOutput * (1.0 - currOutput))
			
			# ---------------------------------------------------------
			# NEU: BIAS ANPASSEN
			# Der Bias lernt mit exakt demselben Gradienten wie die Gewichte!
			# ---------------------------------------------------------
			var neuer_bias = currNeuron.bias + (learningRate * gradient)
			currNeuron.biasWishes.append(neuer_bias)
			
			
			var weightWish = []
			for weightIndex in range(currNeuron.weigths.size()):
				var activation = LAYERS[currLayer-1][weightIndex].currOutput 
				var weight = currNeuron.weigths[weightIndex]
				
				# 2. GEWICHTE ANPASSEN
				var neues_gewicht = weight + (learningRate * gradient * activation)
				weightWish.append(neues_gewicht)
				
				# 3. FEHLER NACH HINTEN WEITERGEBEN
				# Wir addieren die "Schuld" dieses Gewichts auf das Neuron der vorherigen Schicht.
				fehler_fuer_naechste_schicht_rueckwaerts[weightIndex] += gradient * weight
				
			currNeuron.weigthWishes.append(weightWish)
			
		# Wenn wir mit dieser Schicht fertig sind, übergeben wir die gesammelten Fehler
		# an die nächste Runde der Schleife (welche die Schicht davor behandelt).
		fehler_der_aktuellen_schicht = fehler_fuer_naechste_schicht_rueckwaerts
	
func backPropOLD(dataSetIndex) ->void:#ignoring the bias for now
	#to start back propagation the sequenz has to be evaluated by the network once
	pixelgrid.drawNumber(getData(dataSetIndex))
	var currInput = getData(dataSetIndex)
	for i in range(currInput.size()):
		currInput[i] = currInput[i]/255.0
	var expectedTestNumber = getExpectedNumber(dataSetIndex)
	for layer in range(LAYERS.size()):
		var tmpInput = []
		for neuron in range(LAYERS[layer].size()):
			LAYERS[layer][neuron].setIndexInLayer(neuron)
			#print("Layer: "+ str(layer))
			tmpInput.append(LAYERS[layer][neuron].activate(currInput))
		pass
		currInput = tmpInput
	#evaluation done
	#backPropagate trough layers
	var outputForPrevLayer: Array = []
	for currLayer in range(LAYERS.size()-1,0,-1): #counts until 1 (stops before 0 is reached) and first layer has no adjustable weights
	
		
		var wishedOutput = []
		if(currLayer == LAYERS.size()-1): #if its the first iteration the wished output is created by the correkt number
			wishedOutput.resize(10)
			wishedOutput.fill(0.0)
			wishedOutput[expectedTestNumber] = 1.0
		else:#the wished inputs added up from the previous layer are creating the next wishedOutput layer
			for i in range(outputForPrevLayer[0].size()): #all input arrays should be equally long and here iteration from left to right is needed
				var tmp = 0.0
				for j in range(outputForPrevLayer.size()):
					tmp+=outputForPrevLayer[j][i]
					pass
				wishedOutput.append(tmp)
			outputForPrevLayer.clear()
			pass
	
	
		for index in range(wishedOutput.size()):
			if(wishedOutput.size() != LAYERS[currLayer].size()):
				printerr("sizes dont match!")
			#set weightwishes
			var currWish = wishedOutput[index]
			var currNeuron:Neuron = LAYERS[currLayer][index]
			var currOutput = currNeuron.currOutput
			if(currOutput == -1):
				printerr("no run affected this neuron")
			var increaseOutput:bool = currWish>currOutput
			#if increase: positive input-weights should be increased in proportion to their activation and negative weights should be decreased in proportion to their activation
			#if decreas: positive input-weights should be decreased in proportion to their activation and negative weights should be increased in proportion to their activation
			var weightWish = []
			for weightIndex in range(currNeuron.weigths.size()):
				var err = currWish - currOutput
				var activation = LAYERS[currLayer-1][weightIndex].currOutput #is the output of the neuron in the previous layer affecting this weight
				var weight = currNeuron.weigths[weightIndex]
				var neues_gewicht = weight + (learningRate * err * activation)
				weightWish.append(neues_gewicht)
				
				
				
				#if(increaseOutput):
				#	if(weight>0):#increase weight
				#		var proportion = 1-activation #high activation needs less increase
				#		#weightWish.append(weight*(1+proportion))
				#		weightWish.append(weight+(learningRate*proportion))
				#		pass
				#	else:#decrease weight
				#		var proportion = activation #high activation needs stronger decrease
				#		#weightWish.append(weight*(1-proportion))
				#		weightWish.append(weight+(learningRate*proportion))
				#		pass
				#else:#decreaseOutput
				#	if(weight<0):#increase weight
				#		var proportion = 1-activation #high activation needs less increase
				#		#weightWish.append(weight*(1+proportion))
				#		weightWish.append(weight+(learningRate*proportion))
				#		pass
				#	else:#decrease weight
				#		var proportion = activation #high activation needs stronger decrease
				#		#weightWish.append(weight*(1-proportion))
				#		weightWish.append(weight-(learningRate*proportion))
				#		pass
				pass
			#!!!! hier wäre auch der punkt wo man den bias anpassen könnte Neuron.biasWishes !!!
			
			currNeuron.weigthWishes.append(weightWish)
			outputForPrevLayer.append(weightWish)
		pass

func guess(dataSetIndex) -> bool:
	var currInput = getData(dataSetIndex)
	for i in range(currInput.size()):
		currInput[i] = currInput[i]/255.0
	var expectedTestNumber = getExpectedNumber(dataSetIndex)
	for layer in range(LAYERS.size()):
		var tmpInput = []
		for neuron in range(LAYERS[layer].size()):
			LAYERS[layer][neuron].setIndexInLayer(neuron)
			tmpInput.append(LAYERS[layer][neuron].activate(currInput))
		pass
		currInput = tmpInput
	pass
	var guessConfidence = -1
	var guess = -1
	for neuron in range(LAYERS[LAYERS.size()-1].size()):
		var tmpConfidence = LAYERS[LAYERS.size()-1][neuron].currOutput
		if(tmpConfidence>guessConfidence):
			guessConfidence = tmpConfidence
			guess = LAYERS[LAYERS.size()-1][neuron].indexInLayer
		pass
	return (guess == expectedTestNumber)

func cost(dataSetIndex) -> float: 
	pixelgrid.drawNumber(getData(dataSetIndex))
	var currInput = getData(dataSetIndex)
	#normalise pixelvaluse max 255
	#print(currInput.size())
	for i in range(currInput.size()):
		currInput[i] = currInput[i]/255.0
	var expectedTestNumber = getExpectedNumber(dataSetIndex)
	#print(expectedTestNumber)
	
	#iterate trough layers
	for layer in range(LAYERS.size()):
		var tmpInput = []
		for neuron in range(LAYERS[layer].size()):
			LAYERS[layer][neuron].setIndexInLayer(neuron)
			#print("Layer: "+ str(layer))
			tmpInput.append(LAYERS[layer][neuron].activate(currInput))
		pass
		currInput = tmpInput
	pass
	
	#guess is outputneuron with highest output
	var cost = 0
	
	var guessConfidence = -1
	var guess = -1
	for neuron in range(LAYERS[LAYERS.size()-1].size()): #iterate trough last layer

		var tmpConfidence = LAYERS[LAYERS.size()-1][neuron].currOutput
		if(neuron != expectedTestNumber):
			cost += (tmpConfidence-0)*(tmpConfidence-0)
		else:
			cost += (tmpConfidence-1)*(tmpConfidence-1)
		
		if(tmpConfidence>guessConfidence):
			guessConfidence = tmpConfidence
			guess = LAYERS[LAYERS.size()-1][neuron].indexInLayer
		pass
	

	GuessLabel.text = "Guessed: " + str(guess) + "\n with confidence: "+str(guessConfidence) +"\n correct Guess: " + str(expectedTestNumber) + "\n Kosten "+str(cost)
	return cost
	
func _connectLayers() -> void: 
	for i in range(LAYERS.size()):
		for k in range(LAYERS[i].size()):
			var nextLayerIndex = i+1
			var currNeuron:Neuron = LAYERS[i][k]
			if(nextLayerIndex<LAYERS.size()):
				currNeuron.setOutputNeurons(LAYERS[nextLayerIndex])
				#currNeuron.drawLines()
				currNeuron.setInputsNeurons()
				pass
			pass
		pass
	pass

func _draw() -> void:
	for i in range(LAYERS.size() - 1):
		for neuron_a in LAYERS[i]:
			for k in range(neuron_a.outputneurons.size()):
				var neuron_b = neuron_a.outputneurons[k]
				
				# Hole die gespeicherte Farbe für genau diese eine Verbindung!
				var my_color = neuron_a.line_colors[k]
				
				draw_line(neuron_a.position, neuron_b.position, my_color, 0.08)
				#queue_redraw()

func _createLayers() -> void:
	for k in get_children():
		if k is Neuron:
			k.queue_free()
			
	var layer = []
	for i in range(28*28): #first layer		
		var tmpNeuron = NEURON_SZENE.instantiate()
		tmpNeuron.position.x = 330
		tmpNeuron.position.y = 5+(float(i)/(28*28))*(645-5)
		tmpNeuron.scale = Vector2(0.05,0.05)
		tmpNeuron.inFirstLayer = true
		add_child(tmpNeuron)
		layer.append(tmpNeuron)
	LAYERS.append(layer)
	layer = []
	
	for i in range(128):
		var tmpNeuron = NEURON_SZENE.instantiate()
		tmpNeuron.position.x = 450
		tmpNeuron.position.y = 5+(float(i)/(128))*(645-5)
		tmpNeuron.scale = Vector2(0.05,0.05)
		add_child(tmpNeuron)
		layer.append(tmpNeuron)
	LAYERS.append(layer)
	layer = []
	
	for i in range(64):
		var tmpNeuron = NEURON_SZENE.instantiate()
		tmpNeuron.position.x = 570
		tmpNeuron.position.y = 5+(float(i)/(64))*(645-5)
		tmpNeuron.scale = Vector2(0.05,0.05)
		add_child(tmpNeuron)
		layer.append(tmpNeuron)
	LAYERS.append(layer)
	layer = []
		
	for i in range(10):
		var tmpNeuron = NEURON_SZENE.instantiate()
		tmpNeuron.position.x = 690
		tmpNeuron.position.y = 40+(float(i)/(10))*(645-40)
		tmpNeuron.scale = Vector2(0.7,0.7)
		tmpNeuron.inLastLayer = true
		add_child(tmpNeuron)
		layer.append(tmpNeuron)
		
		var label = Label.new()
		label.position.x = 750
		label.position.y = 30+(float(i)/(10))*(645-40)
		label.text = str(i)
		add_child(label)
	LAYERS.append(layer)
	layer = [] 
	pass


func getData(index) -> Array:
	return dataset[index].slice(1)
	pass
func getExpectedNumber(index) -> int:
	return dataset[index][0]
	pass

func load_mnist_database(pfad: String) -> Array:
	var datenbank = []
	var datei = FileAccess.open(pfad, FileAccess.READ)
	
	if not datei:
		print("Datei konnte nicht geöffnet werden!")
		return []

	# Falls die Datei einen Header hat (label, pixel0...), erste Zeile überspringen
	datei.get_line()

	# Zeile für Zeile einlesen
	while not datei.eof_reached():
		var zeile = datei.get_line()
		if zeile == "": continue # Überspringe leere Zeilen am Ende
		
		# Die Zeile am Komma trennen -> ergibt ein Array aus Strings
		var teile = zeile.split(",")
		
		# Wir wandeln die Strings in Zahlen um (Wichtig für Berechnungen!)
		var bild_daten = []
		for wert in teile:
			bild_daten.append(wert.to_int())
			
		# Dieses "Bild-Array" fügen wir unserer Datenbank hinzu
		datenbank.append(bild_daten)
		numbersLoaded=numbersLoaded+1
		if datenbank.size() >= loadMax && loadMax >= 0: break #negative to remove limit
	
	#print("Fertig! Geladene Bilder: ", datenbank.size())
	return datenbank
