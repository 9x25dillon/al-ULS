#!/usr/bin/env julia
"""
Julia dependency installer for TA ULS Training System
"""

using Pkg

println("Installing Julia dependencies for TA ULS Training System...")
println("=" * 60)

# List of required packages
required_packages = [
    "DynamicPolynomials",
    "MultivariatePolynomials", 
    "LinearAlgebra",
    "JSON",
    "Random",
    "HTTP",
    "Statistics"
]

# Install packages
for package in required_packages
    try
        println("Installing $package...")
        Pkg.add(package)
        println("✓ $package installed successfully")
    catch e
        println("✗ Failed to install $package: $e")
    end
end

println("=" * 60)
println("Testing package imports...")

# Test imports
try
    using DynamicPolynomials
    using MultivariatePolynomials
    using LinearAlgebra
    using JSON
    using Random
    using HTTP
    using Statistics
    println("✓ All packages imported successfully")
    println("Julia environment is ready for TA ULS!")
catch e
    println("✗ Import test failed: $e")
    println("Some packages may need manual installation.")
end