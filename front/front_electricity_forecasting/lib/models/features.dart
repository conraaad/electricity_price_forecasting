
import 'dart:convert';

class FeatureResponse {
    String date;
    List<String> featuresNames;
    Map<String, Hour> hours;

    FeatureResponse({
        required this.date,
        required this.featuresNames,
        required this.hours,
    });

    factory FeatureResponse.fromJson(String str) => FeatureResponse.fromMap(json.decode(str));

    String toJson() => json.encode(toMap());

    factory FeatureResponse.fromMap(Map<String, dynamic> json) => FeatureResponse(
        date: json["date"],
        featuresNames: List<String>.from(json["features_names"].map((x) => x)),
        hours: Map.from(json["hours"]).map((k, v) => MapEntry<String, Hour>(k, Hour.fromMap(v))),
    );

    Map<String, dynamic> toMap() => {
        "date": date,
        "features_names": List<dynamic>.from(featuresNames.map((x) => x)),
        "hours": Map.from(hours).map((k, v) => MapEntry<String, dynamic>(k, v.toMap())),
    };
}

class Hour {
    List<double> features;
    double targetPrice;

    Hour({
        required this.features,
        required this.targetPrice,
    });

    factory Hour.fromJson(String str) => Hour.fromMap(json.decode(str));

    String toJson() => json.encode(toMap());

    factory Hour.fromMap(Map<String, dynamic> json) => Hour(
        features: List<double>.from(json["features"].map((x) => x?.toDouble())),
        targetPrice: json["target_price"]?.toDouble(),
    );

    Map<String, dynamic> toMap() => {
        "features": List<dynamic>.from(features.map((x) => x)),
        "target_price": targetPrice,
    };
}
