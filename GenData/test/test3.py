from PIL import Image, ImageOps

background = Image.open("bg/canteen.jpg")
width, height = background.size

foreground = Image.open("crewmate/red.png")
foreground = foreground.resize((width, height))
foreground = foreground.rotate(90, Image.NEAREST, expand=1)
# foreground = ImageOps.flip(foreground)
# foreground = ImageOps.mirror(foreground)

background.paste(foreground, (0, 0), foreground.convert('RGBA'))
background.show()
