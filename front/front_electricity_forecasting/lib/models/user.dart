import 'dart:convert';

class User {
  String email;
  String apiKey;

  User({
    required this.email,
    required this.apiKey,
  });

  // Crea un User a partir d'una cadena JSON
  factory User.fromJson(String source) => User.fromMap(json.decode(source));

  // Crea un User a partir d'un map (objecte deserialitzat)
  factory User.fromMap(Map<String, dynamic> json) {
    return User(
      email: json['email'],
      apiKey: json['api_key'],
    );
  }

  // Converteix el User a un Map
  Map<String, dynamic> toMap() {
    return {
      'email': email,
      'api_key': apiKey,
    };
  }

  // Converteix el User a una cadena JSON
  String toJson() => json.encode(toMap());
}
