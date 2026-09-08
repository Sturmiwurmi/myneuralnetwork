extends Node2D
class_name Neuron



#logic
var bias:float = 0.0
@export var biasWishes: Array[float] = [] 
var weigths:Array[float] = [] #gemini: professional variant: Xavier/Glorot initialization -> value range depends on the number of inputs a neuron has: ± 1/sqrt(n)
var indexInLayer = -1
@export var currOutput = -1
@export var weigthWishes:Array = []

func applyWeightWishes() -> void:
	
	if(!inFirstLayer):
		if biasWishes.size() > 0:
			var bias_tmp = 0.0
			for b in biasWishes:
				bias_tmp += b
			bias = bias_tmp / biasWishes.size()
		var count = weigthWishes.size()
		for i in range(weigthWishes[0].size()):
			var tmp = 0.0
			for j in range(weigthWishes.size()):
				tmp+=weigthWishes[j][i]
			tmp = tmp/count
			weigths[i] = tmp
			#print(tmp)
		pass
	weigthWishes.clear()
	biasWishes.clear()
#randfn()
func initWeigths() -> void: #has to be called when network is fully instantiated
	if(!inFirstLayer):#first layer has no relevant weights
		for i in range(inputneurons.size()):
			weigths.append(randfn(0.0,1/sqrt(inputneurons.size()))) #normally distributed with Xavier/Glorot  initialization
			pass
		pass

func activate(prevOutputArr) -> float: 
	if(inFirstLayer):
		my_sprite.self_modulate = Color(colorHex)
		my_sprite.self_modulate.s = prevOutputArr[indexInLayer]
		return prevOutputArr[indexInLayer]
		pass
	else:
		if(prevOutputArr.size() != inputneurons.size()):
			
			print(str(prevOutputArr.size()) + " given input")
			print(str(inputneurons.size()) + " expected input")
			printerr("WRONG INPUT SIZE")
	
		var output = bias
		
		for i in range(inputneurons.size()):
			output += prevOutputArr[i] * weigths[i]
			pass		
		output = sigmoid(output)
		currOutput = output
		my_sprite.self_modulate = Color(colorHex)
		my_sprite.self_modulate.s = output
		return output
			
	pass

func sigmoid(x) -> float:
	return 1 / (1 + exp(x * -1))
	pass

func setIndexInLayer(x) -> void: 
	indexInLayer = x
	pass
#visual
@export var outputneurons:Array[Neuron] = [] # next layer
@export var inputneurons:Array[Neuron] = [] # input layer (automatically assigned)
@export var inFirstLayer:bool = false
@export var inLastLayer:bool = false
var lines:Array[Line2D] = []
@onready var my_sprite = $Sprite2D2
@export var colorHex:String = "#00cce3"
var line_colors: Array[Color] = []

# Called when the node enters the scene tree for the first time.
func _ready() -> void:	
	
	
	outputneurons.erase(self) #error correction
	drawLines()
	
	my_sprite.self_modulate.s = 0
	
	for i in range(lines.size()):
		lines[i].default_color.s = 0 
		pass
	
	setInputsNeurons()
	pass # Replace with function body.



func setInputsNeurons() -> void: #!! to set Inputs from OutputNeurons. Not the inputs from this neuron
	for i in range(outputneurons.size()):
		outputneurons[i].inputneurons.append(self)
	pass

func setOutputNeurons(arr) -> void:
	outputneurons.assign(arr)	
	line_colors.clear()
	for i in range(outputneurons.size()):
		line_colors.append(Color.CYAN)
	pass

func drawLines() -> void: 
	for i in range(outputneurons.size()):
		var myline2d:Line2D = Line2D.new()
		myline2d.z_index = -1
		myline2d.width = 1.0
		myline2d.default_color = Color.CYAN
		#myline2d.antialiased = true
		myline2d.add_point(Vector2(0,0))
		# Endpunkt (Automatisch perfekt umgerechnet, egal welcher Scale!)
		myline2d.add_point(to_local(outputneurons[i].global_position))
		lines.append(myline2d)
		add_child(myline2d)		
	pass


func _process(delta: float) -> void:
	pass
