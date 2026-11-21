module class1_tree15(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[253]) & (~i[78])) | (i[296] & i[253] & (~i[78])) | (i[43] & i[211] & i[365] & i[78]) | ((~i[40]) & (~i[211]) & i[365] & i[78]) | (i[82] & i[274] & (~i[365]) & i[78]) | (i[361] & (~i[274]) & (~i[365]) & i[78]) | (i[146] & (~i[43]) & i[211] & i[365] & i[78]) | ((~i[81]) & i[40] & (~i[211]) & i[365] & i[78]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
