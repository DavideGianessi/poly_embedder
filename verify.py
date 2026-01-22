import sys

p = 4194304001

def read_input_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    n, degree = map(int, lines[0].strip().split())
    points = []
    for i in range(1, n + 1):
        x, y = map(int, lines[i].strip().split())
        points.append((x, y))
    return n, degree, points

def read_output_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    num_coeffs = int(lines[0].strip())
    coefficients = []
    for i in range(1, num_coeffs + 1):
        coeff = int(lines[i].strip())
        coefficients.append(coeff)
    return num_coeffs, coefficients

def evaluate_polynomial(coefficients, x):
    result = 0
    for coeff in reversed(coefficients):
        result = (result * x + coeff) % p
    return result

def check_polynomial():
    try:
        n_points, required_degree, points = read_input_file('input.txt')
        num_coeffs, coefficients = read_output_file('output.txt')
        print(f"Input file analysis:")
        print(f"  Number of points: {n_points}")
        print(f"  Required polynomial degree: {required_degree}")
        print(f"\nOutput file analysis:")
        print(f"  Number of coefficients: {num_coeffs}")
        print(f"  Actual polynomial degree: {num_coeffs - 1}")
        if num_coeffs - 1 != required_degree:
            print(f"\n❌ ERROR: Degree mismatch!")
            print(f"   Required: {required_degree}, but got polynomial of degree {num_coeffs - 1}")
            return False
        else:
            print(f"\n✓ Degree check passed: {required_degree} == {num_coeffs - 1}")
        print(f"\nChecking coefficients range [0, {p-1}]:")
        all_in_range = True
        for i, coeff in enumerate(coefficients):
            if coeff < 0 or coeff >= p:
                print(f"  ❌ Coefficient {i} (degree {i}): {coeff} is out of range!")
                all_in_range = False
        if all_in_range:
            print("  ✓ All coefficients are in valid range")
        else:
            print("  ❌ Some coefficients are out of range")
            return False
        print(f"\nEvaluating polynomial at {n_points} points:")
        all_points_match = True
        for i, (x, expected_y) in enumerate(points):
            computed_y = evaluate_polynomial(coefficients, x)
            expected_y_mod = expected_y % p
            computed_y_mod = computed_y % p
            if expected_y_mod != computed_y_mod:
                print(f"  ❌ Point {i}: P({x})")
                print(f"     Expected: {expected_y} (mod p = {expected_y_mod})")
                print(f"     Computed: {computed_y} (mod p = {computed_y_mod})")
                all_points_match = False
            else:
                print(f"  ✓ Point {i}: P({x}) = {expected_y} (mod p = {expected_y_mod})")
        if all_points_match:
            print("\n✓ All points satisfy the polynomial!")
            return True
        else:
            print("\n❌ Some points do not satisfy the polynomial!")
            return False
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        return False
    except ValueError as e:
        print(f"❌ Error: Invalid data format - {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main execution function."""
    print("=" * 60)
    print("POLYNOMIAL VERIFICATION SCRIPT")
    print(f"Field modulus: p = {p}")
    print("=" * 60)
    success = check_polynomial()
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL CHECKS PASSED")
    else:
        print("❌ VERIFICATION FAILED")
    print("=" * 60)

if __name__ == "__main__":
    main()
