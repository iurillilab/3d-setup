from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


def generate_vector_checkerboard_pdf(square_size_cm, num_squares_y, num_squares_x, 
                                     filename, topleft="white"):
    """Generate a vector checkerboard PDF with a circle above the top left square.

    Parameters
    ----------
    square_size_cm : float or int
        Size of each square of the checkerboard in centimeters
    num_squares_y : int
        Number of squares in the vertical direction
    num_squares_x : int
        Number of squares in the horizontal direction
    filename : str or Path
        Filename to save the generated PDF
    topleft : str, optional
        Color of the top left square, by default "white"
    """
    # Convert square size from cm to points
    square_size = square_size_cm * cm  # ReportLab uses cm as a unit conversion factor
    
    # Initialize the PDF canvas with the specified filename and A4 page size
    c = canvas.Canvas(str(filename), pagesize=A4)
    page_width, page_height = A4
    
    # Calculate the total width and height of the checkerboard
    checkerboard_width = square_size * num_squares_x
    checkerboard_height = square_size * num_squares_y
    
    # Calculate offsets to center the checkerboard on the page
    offset_x = (page_width - checkerboard_width) / 2
    offset_y = (page_height - checkerboard_height) / 2
    
    # Draw the checkerboard using vector graphics
    assert topleft in ["white", "black"], "Invalid specification for top left square"
    off = 1 if topleft == "white" else 0
    for y in range(num_squares_y):
        for x in range(num_squares_x):
            if (x + y) % 2 + off == 1:  # Fill the square if it's part of the checker pattern
                c.rect(offset_x + x * square_size, offset_y + y * square_size, square_size, square_size, fill=1)
    
    # Calculate the center of the top left square for the circle
    circle_center_x = offset_x + square_size / 2
    circle_center_y = offset_y + square_size / 2 + checkerboard_height

    # Draw the circle above the checkerboard, half inside and half outside the top left square
    c.circle(circle_center_x, circle_center_y, square_size / 4, fill=1)
    
    # Finalize and save the PDF
    c.showPage()
    c.save()


# Example usage
generate_vector_checkerboard_pdf(1, 8, 11, 
                                 "/Users/vigji/Desktop/vector_checkerboard.pdf",
                                 topleft="white")