"""

A tutorial for solving PDE's in Julia !

    julia 1.9.4

"""

"""

Finite difference method for the heat equation

    ∂ₜ u - ε ∂ₓₓ u = f(x,t,u)
    u(0,x) = u₀(x)
    u(t,a) = uₗ(x)
    u(t,b) = uᵣ(x)
    ∂ₜ U = -M U  + F(t,U) = S(t,U)

    ex :

    u = exp(x-t)
    ∂ₓu = exp(x-t)
    ∂ₓₓ u =exp(x-t)
    ∂ₜ u = -exp(x-t)
    S = exp(x-t) ( ε-1)
"""

using DifferentialEquations
using LinearAlgebra,SparseArrays
using Symbolics

# define the problem

@inline uex(x,t) = exp(x-t)
@inline u₀(x) = exp(x)
@inline uₗ(t) = exp(-t)
@inline uᵣ(t) = exp(1-t)
@inline f(x,t,u) = exp(x-t)*( 1e-2 - 1)

function S!(Up,U,p,t)
    M = p[1]
    x = p[2]
    h= p[3]
    ε = p[4]
    F = [ -f(xi,t,U[i]) for (i,xi) in enumerate(x) ]
    F[1] -= ε*uₗ(t)/h^2
    F[end] -= ε*uᵣ(t)/h^2
    Up .= - muladd(M,U,F)    # ∂ₜ U = - (M U  +  (-F) )
end

function main(N;ε = 1e-2)
    a = 0
    b = 1
    h = (b-a)/(N+1) 
    x = a+h:h:b-h
    O = -ones(N-1)./h^2
    M = ε.*spdiagm(0 => 2/h^2*ones(N), -1=> O, 1=> O) # This is the basic approximation of the second derivative
    p = (M,x,h,ε)
    U0 = u₀.(x)
    tspan = (0,3) # (t0, tfinal)
    jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> S!(du, u, p, 0.0),
    copy(U0), U0) # get the spacity of the problem can help a lot the solver
    display(jac_sparsity)
    f = ODEFunction(S!,jac_prototype = float.(jac_sparsity)) # adapt the function to use the spacity
    prob = ODEProblem(f, U0, tspan,p) # create the ODE problem
    sol = solve(prob,reltol = 1e-8, abstol = 1e-8)   # Solve (Here, we let him choose the solver)
    x,sol
end


@time x,sol = main(100);

uvals = sol.u

tvals = sol.t
using Plots

@gif for (i,u) in enumerate(uvals)
    plot(x,u,title=tvals[i],lw=3,ylim = (0,3))
    plot!(x,uex.(x,tvals[i]),lw=3)
end

"""

DG method for hyperbolic scalar equation

∂ₜu + ∂ₓ f(u) = s(u)

M ∂ₜ U = S(U) + F(U) = SS(U)

M = h/2 ∫₋₁¹ ϕ ϕᵗ
S = h/2 ∫₋₁¹ s(u)ϕ
F = ∫₋₁¹ f(u)∂ₓ ϕ - f̂( u(x_{n+1}^-),u(x_{n+1}^+) ) ϕ(1) + f̂( u(x_n^-),u(x_n^+) ) ϕ(-1)

"""

using DifferentialEquations
using QuadGK,LinearAlgebra

# define the problem 
xx(y,el) = el.h/2*y + el.c 
yy(x,el) = 2/el.h*(x-el.c)
u0(x) = exp(-50*(x-0.5)^2)
f(x,t,u) = u^2/2 
fp(x,t,u) = u
s(x,t,u) = 0.0

# define the boundary conditions (take care of your ability to impose a value)
function ug(x,t,ugg) 
    if fp(x,t,ugg) < 0 # at the inlet, the characteristic is sorting
        return ugg # Ugg is the value of the solution at the last point (this is a approximation of ∂ₜ( u(X_0(t),t) ) = s(u(X_0(t),t)) assuming the integral of $s$ small)
    else                # at the inlet, the characteristic is entering
        return 0.0      # we impose a value 0 here (must be coherent with your initial condition)
    end
 end
 function ud(x,t,udd) 
    if fp(x,t,udd) > 0 # at the outlet, the characteristic is sorting
        return udd  
    else               # at the outlet, the characteristic is entering
        return 0.0
    end
 end

# define the polynomial bases used
ϕ(y,i) = y^i
ϕp(y,i) = i==0 ? 0 : i*y^(i-1) 
# define the numerical flux (here Lax-Friedrish)
f̂(ug,ud,x,t) = 0.5*(f(x,t,ug) + f(x,t,ud) - max(abs(fp(x,t,ug)),abs(fp(x,t,ud)))*(ud-ug))

function SS!(Up,U,p,t)
    l = p[1]
    w = p[2]
    deg = p[3]
    h = p[4]
    c = p[5]
    M = p[6]
    res = zeros(deg+1)
    for n in axes(U,1)
        res .= 0.0
        for j in 0:deg
            # calculate the integral terms
            for i in eachindex(l)
                    ui = sum(ϕ(l[i],k-1)*U[n,k] for k in axes(U,2))
                    res[j+1] += w[i]* (h/2*s(xx(l[i],(h=h,c=c[n])),t,ui)*ϕ(l[i],j) + f(xx(l[i],(h=h,c=c[n])),t,ui)*ϕp(l[i],j) )
            end
            # set the different trace of the function taking care of if we are a a boundary
            ugp = 0.0
            udm = 0.0
            ugm = 0.0
            udp = 0.0
            if n>1 && n<size(U,1) 
                for k in axes(U,2)
                    ugp += ϕ(-1,k-1)*U[n,k]
                    udm += ϕ(1,k-1)*U[n,k]
                    ugm += ϕ(1,k-1)*U[n-1,k]
                    udp += ϕ(-1,k-1)*U[n+1,k]
                end
            elseif n==1
                for k in axes(U,2)
                    ugp += ϕ(-1,k-1)*U[n,k]
                    udm += ϕ(1,k-1)*U[n,k]
                    udp += ϕ(-1,k-1)*U[n+1,k]
                end
                ugm = ug(c[n]+h/2,t,udm)
            elseif n==size(U,1)
                for k in axes(U,2)
                    ugp += ϕ(-1,k-1)*U[n,k]
                    udm += ϕ(1,k-1)*U[n,k]
                    ugm += ϕ(1,k-1)*U[n-1,k]
                end
                udp = ud(c[end]-h/2,t,ugp)
            end
            # use the Lax Friedrish Flux
            res[j+1] += f̂(ugm,ugp,c[n]-h/2,t)*ϕ(-1,j) - f̂(udm,udp,c[n]+h/2,t)*ϕ(1,j)
        end
        #  solve M ∂ₜ U = res "localy" on this element and put them in Up
        ldiv!(M,res)
        for i in axes(Up,2)
            Up[n,i] = res[i]
        end
    end
    nothing
end


function main(N,deg)
    a = 0
    b = 1
    h = (b-a)/N 
    c = a+h/2:h:b-h/2
    l,w = gauss(2*deg+1) # make sur the integral approximation are exact in the linear case (f(u)=u) -> integrate a 2*deg order polynomial -> 2*deg+1 order for gauss
    M = zeros(deg+1,deg+1)
    for i in 0:deg
        for j in 0:deg 
            for k in eachindex(l)
                M[i+1,j+1] += h/2 * w[k] * ϕ(l[k],i) * ϕ(l[k],j) 
            end
        end
    end
    M = factorize(M) # store the factorization instead of the matrix since M is only used to solve systems
    # Calculate the degree of freedom of the initial condition in each element  ( M U₀ = ∫₋₁¹ u0( y^{-1}(y) ) ϕ( y ) )
    U0 = zeros(N,deg+1)
    V = zeros(deg+1)
    for n in axes(U0,1)
        V.=0
        for i in 0:deg
            for k in eachindex(l)
                V[i+1] += h/2 * w[k] * u0(xx(l[k],(h=h,c=c[n])))*ϕ(l[k],i)
            end
        end
        U0[n,:] .= (M\V)
    end
    p=(l,w,deg,h,c,M)
    tspan = (0.0,1.0)
    # again define the ODE problem ( We don't use spacity since we used a "local" method )
    prob = ODEProblem(SS!, U0, tspan,p)
    # The method is of order deg+1 as long as the time-order method is also deg+1 or more, we then specify algorithm for each case
    alg = begin
        if deg ==0
            Heun() # we use Heun instead of Euler because we don't want to have to specify a dt
        elseif deg ==1
            Heun()
        elseif deg==2 || deg==3
            RK4()
        elseif deg ==4
            DP5()
        elseif deg ==5 || deg==6
            TanYam7()
        else deg ==7
            TsitPap8()
        end
    end
    # We solve the problem saving every 0.01 time
    @time sol = solve(prob,alg,saveat=0.01)   
    uu = sol.u
    tt = sol.t
    uvals = Vector{Float64}[]
    for i in eachindex(tt)
        ut = Float64[]
        for j in axes(uu[i],1)
            for k in eachindex(l)
                push!(ut,sum( ϕ(l[k],ideg-1)*uu[i][j,ideg] for ideg in axes(uu[i],2) ))
            end
        end
        push!(uvals,ut)
    end
    xvals = Float64[]
    for n in axes(U0,1)
        for i in eachindex(l)
            push!(xvals,xx(l[i],(h=h,c=c[n])))
        end
    end
    xvals,uvals,tt
end

@time x,uvals,tvals = main(100,1);

using Plots
@gif for (i,u) in enumerate(uvals)
    plot(x,u,title="$(tvals[i])",ylim=(-0.5,1.5))
end

"""

Coming soon : 
- add callback to avoid oscilation
- add the system case
- add an advection-diffusion scalar case

"""