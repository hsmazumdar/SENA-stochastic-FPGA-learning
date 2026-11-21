module class1_tree7(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[266]) & i[78]) | ((~i[149]) & i[9] & (~i[78])) | (i[250] & i[149] & i[9] & (~i[78])) | ((~i[144]) & (~i[85]) & i[47] & i[266] & i[78]) | (i[384] & i[126] & (~i[47]) & i[266] & i[78]) | ((~i[419]) & (~i[126]) & (~i[47]) & i[266] & i[78]) | ((~i[119]) & (~i[250]) & i[149] & i[9] & (~i[78])))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
