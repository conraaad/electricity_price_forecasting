// ignore_for_file: use_build_context_synchronously

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:front_electricity_forecasting/providers/auth_provider.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>(); // Move here

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<AuthProvider>(context);

    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            const SizedBox(height: 100),
            const Text(
              'Benvingut a l\'aplicació de prediccions del preu elèctric',
              style: TextStyle(
                fontSize: 50,
                fontWeight: FontWeight.bold,
                color: Colors.black,
              ),
            ),

            const SizedBox(height: 180),
            
            Container(
              width: 600,
              padding: const EdgeInsets.all(32),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: const [
                  BoxShadow(
                    color: Colors.black54,
                    blurRadius: 15,
                    offset: Offset(0, 4),
                  )
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [

                  const Text(
                    'Registra el teu correu',
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: Colors.black,
                    ),
                  ),
                  
                  const SizedBox(height: 24),

                  const Text(
                    'Per poder fer ús de l\'aplicació, si us plau, registra el teu correu electrònic. Aquest registre també proporciona accés a l\'API de prediccions.',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.black,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  
                  const SizedBox(height: 24),

                  Form(
                    key: _formKey, // Use local key
                    autovalidateMode: AutovalidateMode.disabled,
                    child: TextFormField(
                      decoration: InputDecoration(
                        labelText: 'Correu electrònic',
                        labelStyle: const TextStyle(
                          color: Colors.black,
                          fontSize: 16,
                        ),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                          borderSide: const BorderSide(
                            color: Colors.black,
                            width: 2,
                          ),
                        ),
                        focusedBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                          borderSide: const BorderSide(
                            color: Colors.black,
                            width: 2,
                          ),
                        ),
                        prefixIcon: const Icon(Icons.email_outlined, color: Colors.black, size: 24),
                      ),
                      keyboardType: TextInputType.emailAddress,
                      onChanged: (value) => provider.email = value,
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return 'Si us plau, introdueix el teu correu electrònic';
                        }
                        final emailPattern = RegExp(r'^[^@]+@[^@]+\.[^@]+$');
                        if (!emailPattern.hasMatch(value)) {
                          return 'Si us plau, introdueix un correu electrònic vàlid';
                        }
                        return null;
                      },
                    ),
                  ),

                  const SizedBox(height: 24),

                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: provider.isLoading 
                        ? null 
                        : () async {
                            if (!(_formKey.currentState?.validate() ?? false)) return;
                            provider.isLoading = true;
                            
                            if (await provider.login()) {
                              // print('Login successful: ${provider.user!.email}');
                              // Navigate to the home screen if login is successful
                              Navigator.pushReplacementNamed(context, 'home');
                            } 
                            else {
                              // Show an error message if login fails
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  backgroundColor: Colors.red,
                                  content: Center(
                                    child: Text(
                                      'Error al registrar-se. Si us plau, intenta-ho de nou.',
                                      style: TextStyle(fontSize: 16, color: Colors.white)
                                    ),
                                  ),
                                )
                              );
                            }
                            provider.isLoading = false;
                          },
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 20),
                        backgroundColor: Colors.black,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: provider.isLoading
                          ? const CircularProgressIndicator(
                              color: Colors.white,
                              strokeWidth: 2,
                            )
                          : const Text(
                              'Registrar-se',
                              style: TextStyle(fontSize: 20, color: Colors.white, fontWeight: FontWeight.w600),
                            ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
