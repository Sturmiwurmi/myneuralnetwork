extends Node2D
class_name Pixelgrid



func _ready() -> void:
	#for i in range(dataset.size()):
	#	erstelle_pixel_bild(dataset[i].slice(1))
	#	$Label.text = str(dataset[i][0])
	#	await get_tree().create_timer(delay).timeout
	pass # Replace with function body.



func drawNumber(pixels:Array) -> void:
	$Label.text = str(pixels[0])
	erstelle_pixel_bild(pixels.slice(1))
	
	pass


func erstelle_pixel_bild(pixel_daten: Array):
	# 1. Ein leeres Bild erstellen (28x28)
	var img = Image.create(28, 28, false, Image.FORMAT_L8)
	
	# 2. Pixel setzen
	for i in range(pixel_daten.size()):
		var x = i % 28
		var y = i / 28
		var wert = float(pixel_daten[i]) / 255.0
		img.set_pixel(x, y, Color(wert, wert, wert))
	
	# 3. Bild in eine Textur umwandeln und einem Sprite zuweisen
	var tex = ImageTexture.create_from_image(img)
	$Sprite2D.texture = tex
	$Sprite2D.scale = Vector2(10, 10) # Damit man die 28x28 Pixel auch sieht
	
