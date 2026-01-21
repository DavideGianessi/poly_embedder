import sys
import random
import secrets

# Field modulus
p = 18446744073692774401

def generate_input_file(num_points, degree):
    """
    Generate an input.txt file with random points.
    
    Args:
        num_points: Number of points to generate
        degree: Required polynomial degree
    
    Returns:
        True if successful, False otherwise
    """
    
    # Validate inputs
    if num_points <= 0:
        print(f"Error: Number of points must be positive, got {num_points}")
        return False
    
    if degree < 0:
        print(f"Error: Degree must be non-negative, got {degree}")
        return False
    
    # We need at least degree+1 points to uniquely determine a polynomial of that degree
    if num_points < degree + 1:
        print(f"Warning: With only {num_points} points, a polynomial of degree {degree}")
        print(f"         may not be uniquely determined (need at least {degree+1} points).")
        print("         Continuing anyway...")
    
    # Generate unique x values
    # Use a set to ensure uniqueness
    x_values = set()
    
    # We might need to try multiple times if we get collisions
    max_attempts = num_points * 10  # Safety limit
    attempts = 0
    
    while len(x_values) < num_points and attempts < max_attempts:
        # Generate random x in the field [0, p-1]
        x = secrets.randbelow(p)
        x_values.add(x)
        attempts += 1
    
    if len(x_values) < num_points:
        print(f"Error: Could not generate {num_points} unique x values after {max_attempts} attempts")
        return False
    
    # Convert to list
    x_list = list(x_values)
    
    # For testing purposes, we could generate y values from a random polynomial
    # But for this script, we'll generate random y values within the field
    # (In practice, these y values would come from evaluating some polynomial)
    
    with open('input.txt', 'w') as f:
        # Write first line: number of points and required degree
        f.write(f"{num_points} {degree}\n")
        
        # Write each point (x, random y)
        for x in x_list:
            y = secrets.randbelow(p)
            f.write(f"{x} {y}\n")
    
    print(f"Successfully generated input.txt with {num_points} points")
    print(f"Required polynomial degree: {degree}")
    print(f"Field modulus: {p}")
    print(f"\nFirst few points:")
    
    # Display first few points for verification
    with open('input.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:min(6, len(lines))]):
            if i == 0:
                print(f"  Header: {line.strip()}")
            else:
                print(f"  Point {i}: {line.strip()}")
    
    if num_points > 5:
        print(f"  ... and {num_points - 5} more points")
    
    return True

def main():
    """Main execution function."""
    
    print("=" * 60)
    print("INPUT FILE GENERATOR")
    print(f"Field modulus: p = {p}")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python generate_input.py <num_points> <degree>")
        print("\nExample: python generate_input.py 5 15")
        print("         Generates 5 points for a polynomial of degree 15")
        print("\nNote: For a polynomial of degree d to be uniquely determined,")
        print("      you need at least d+1 points.")
        return
    
    try:
        num_points = int(sys.argv[1])
        degree = int(sys.argv[2])
    except ValueError:
        print("Error: Both arguments must be integers")
        print(f"Received: num_points='{sys.argv[1]}', degree='{sys.argv[2]}'")
        return
    
    # Generate the input file
    success = generate_input_file(num_points, degree)
    
    print("\n" + "=" * 60)
    if success:
        print("✓ INPUT FILE GENERATED SUCCESSFULLY")
        print("  File saved as: input.txt")
    else:
        print("❌ FAILED TO GENERATE INPUT FILE")
    print("=" * 60)

if __name__ == "__main__":
    main()
