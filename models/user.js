import mongoose from "mongoose";

// Define the User schema
const userSchema = new mongoose.Schema({

    userType: {
        type: String,
        required: [true, 'Holder name is required !'],
    },
    username: {
        type: String,
        required: [true, 'Username is required !'],
    },
    email: {
        type: String,
        required: [true, 'Email is required !'],
    },
    password: {
        type: String,
        required: [true, 'Email is required !'],
    }
});

const User = mongoose.models.user || mongoose.model("Users", userSchema);
export default User