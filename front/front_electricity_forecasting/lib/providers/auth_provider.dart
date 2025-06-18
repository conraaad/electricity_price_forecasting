
import 'dart:convert';

import 'package:flutter/material.dart';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;

import 'package:front_electricity_forecasting/models/user.dart';

class AuthProvider extends ChangeNotifier {

  final FlutterSecureStorage _storage = FlutterSecureStorage();

  // TheProvider() {
  //   loadUserFromPrefs();
  // }

  final String baseUrl = 'http://127.0.0.1:8000/api/v1';

  User? user;

  String email = '';
  bool _isLoading = false;

  bool get isLoading => _isLoading;

  set isLoading(bool value) {
    _isLoading = value;
    notifyListeners();
  }

  Future<void> loadUserFromPrefs() async {
    final json = await _storage.read(key: 'user');
    if (json != null) {
      user = User.fromJson(json);
      notifyListeners();
    }
  }

  Future<bool> login() async {
    final Uri url = Uri.parse('$baseUrl/register');

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'email': email}),
      );

      if (response.statusCode == 201) {
        
        user = User.fromJson(response.body);
        await _storage.write(key: 'user', value: user!.toJson());

        return true;
      } else {
        print('Error: ${response.statusCode}, ${response.body}');
        return false;
      }
    } catch (error) {
      print('Error during login: $error');
      return false;
    }
  }

  Future<void> logout() async {
    user = null;
    await _storage.deleteAll();
    email = '';
    notifyListeners();
  }
}
